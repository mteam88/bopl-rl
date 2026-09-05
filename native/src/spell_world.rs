//! Dynamic spell instances share Updater node/component identities with actors and effects.
use crate::{fixed::Fix, scheduler, shockwave, vector::Vec2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum Prefab {
    Shockwave(shockwave::Config),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "state", rename_all = "snake_case")]
pub enum State {
    Shockwave(shockwave::State),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub prefab: String,
    pub actor: usize,
    pub node: usize,
    pub components: [usize; 2],
    pub position: Vec2,
    pub present: bool,
    pub destroyed: bool,
    pub state: State,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct World {
    pub prefabs: BTreeMap<String, Prefab>,
    pub instances: Vec<Instance>,
}
impl World {
    pub fn spawn(
        &mut self,
        schedule: &mut scheduler::State,
        prefab: &str,
        actor: usize,
        player: i32,
        position: Vec2,
        scale: Fix,
    ) -> scheduler::Result<()> {
        let config = self.prefabs.get(prefab).ok_or("spell_prefab_not_bound")?;
        let node = schedule.nodes.len();
        schedule.nodes.push(scheduler::Node {
            parent: None,
            children: vec![],
            active: true,
            present: true,
        });
        let first = schedule.components.len();
        let components = [first, first + 1];
        schedule.begin_prefab();
        for (i, &id) in components.iter().enumerate() {
            schedule.components.push(scheduler::Component {
                owner: node,
                kind: if i == 0 {
                    scheduler::Kind::Updatable
                } else {
                    scheduler::Kind::Fix
                },
                present: true,
                destroyed: false,
                hierarchy: if i == 0 { 1 } else { 0 },
                enabled: true,
                ignore_activity: false,
                execution_gate: None,
            });
            if i == 0 {
                schedule.register(Some(id));
            }
        }
        schedule.end_prefab()?;
        let state = match config {
            Prefab::Shockwave(_) => State::Shockwave(shockwave::State::new(player, scale)),
        };
        self.instances.push(Instance {
            prefab: prefab.into(),
            actor,
            node,
            components,
            position,
            present: true,
            destroyed: false,
            state,
        });
        Ok(())
    }
    pub fn phase(
        &mut self,
        schedule: &mut scheduler::State,
        targets: &mut shockwave::Targets,
        phase: scheduler::Event,
        dt: Fix,
    ) -> scheduler::Result<bool> {
        let Some(id) = phase.id else {
            return Ok(false);
        };
        if matches!(phase.kind, scheduler::Phase::DestroyObject) {
            for instance in &mut self.instances {
                if instance.node == id {
                    instance.present = false;
                }
            }
            return Ok(false);
        }
        let Some(instance) = self
            .instances
            .iter_mut()
            .find(|v| v.components.contains(&id))
        else {
            return Ok(false);
        };
        let mut destroy = false;
        match (&mut instance.state, &self.prefabs[&instance.prefab]) {
            (State::Shockwave(state), Prefab::Shockwave(config)) => {
                if matches!(phase.kind, scheduler::Phase::Init) && id == instance.components[0] {
                    state.time = Fix::ZERO;
                    destroy = state.activate(config, instance.position, Fix::ZERO, targets)?;
                } else if matches!(phase.kind, scheduler::Phase::Update)
                    && id == instance.components[0]
                {
                    if config.every_frame {
                        destroy = state.update(config, instance.position, dt, targets)?;
                    }
                    destroy |= config.kill_after_frame;
                }
            }
        }
        if destroy {
            schedule.destroy_object(Some(instance.node))?;
            instance.destroyed = true;
        }
        Ok(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Command {
    Activate { dt: Fix },
    Ignore { object: u32 },
    Scale { value: Fix },
    Destroy,
}
impl World {
    pub fn command(
        &mut self,
        index: usize,
        command: Command,
        schedule: &mut scheduler::State,
        targets: &mut shockwave::Targets,
    ) -> scheduler::Result<()> {
        let instance = self.instances.get_mut(index).ok_or("index_out_of_range")?;
        let mut destroy = false;
        match (&mut instance.state, &self.prefabs[&instance.prefab]) {
            (State::Shockwave(state), Prefab::Shockwave(config)) => match command {
                Command::Activate { dt } => {
                    destroy = state.activate(config, instance.position, dt, targets)?
                }
                Command::Ignore { object } => {
                    state.ignored.insert(object);
                }
                Command::Scale { value } => state.scale = value,
                Command::Destroy => destroy = true,
            },
        }
        if destroy {
            schedule.destroy_object(Some(instance.node))?;
            instance.destroyed = true;
        }
        Ok(())
    }
}
