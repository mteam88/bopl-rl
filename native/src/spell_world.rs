//! Dynamic spell instances share Updater node/component identities with actors and effects.
use crate::{fixed::Fix, revival, scheduler, shockwave, vector::Vec2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum Prefab {
    Shockwave(shockwave::Config),
    Marker(revival::Config),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "state", rename_all = "snake_case")]
pub enum State {
    Shockwave(shockwave::State),
    Marker(Box<revival::Marker>),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub prefab: String,
    pub actor: usize,
    pub node: usize,
    pub components: Vec<usize>,
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
        request: Spawn,
    ) -> scheduler::Result<usize> {
        let config = self
            .prefabs
            .get(request.prefab)
            .ok_or("spell_prefab_not_bound")?;
        let marker = matches!(config, Prefab::Marker(_));
        let node = schedule.nodes.len();
        schedule.nodes.push(scheduler::Node {
            parent: None,
            children: vec![],
            active: true,
            present: true,
        });
        let first = schedule.components.len();
        let components: Vec<_> = (first..first + if marker { 4 } else { 2 }).collect();
        schedule.begin_prefab();
        for (i, _) in components.iter().enumerate() {
            schedule.components.push(scheduler::Component {
                owner: node,
                kind: if i == 1 {
                    scheduler::Kind::Fix
                } else {
                    scheduler::Kind::Updatable
                },
                present: true,
                destroyed: false,
                hierarchy: match i {
                    0 => 1,
                    1 => 0,
                    _ => i as i32,
                },
                enabled: true,
                ignore_activity: false,
                execution_gate: None,
            });
        }
        // Original Awake order is BoplBody, DPhysicsCircle, then marker.
        for &i in if marker { &[2, 3, 0][..] } else { &[0][..] } {
            schedule.register(Some(components[i]));
        }
        schedule.end_prefab()?;
        let state = match config {
            Prefab::Shockwave(_) => {
                State::Shockwave(shockwave::State::new(request.player, request.scale))
            }
            Prefab::Marker(c) => State::Marker(Box::new(revival::Marker::new(
                c,
                request.object.ok_or("missing_spell_object_id")?,
                request.player,
                request.position,
            ))),
        };
        let index = self.instances.len();
        self.instances.push(Instance {
            prefab: request.prefab.into(),
            actor: request.actor,
            node,
            components,
            position: request.position,
            present: true,
            destroyed: false,
            state,
        });
        Ok(index)
    }
    pub fn end_marker(
        &mut self,
        index: usize,
        schedule: &mut scheduler::State,
        scene: &mut crate::physics::scene::Scene,
    ) -> scheduler::Result<()> {
        let instance = self.instances.get_mut(index).ok_or("index_out_of_range")?;
        let (State::Marker(marker), Prefab::Marker(config)) =
            (&mut instance.state, &self.prefabs[&instance.prefab])
        else {
            return Err("marker_required");
        };
        if !schedule.components[instance.components[0]].present {
            return Ok(());
        }
        if config.particle {
            for i in [0, 2, 3, 1] {
                schedule.destroy_mono(Some(instance.components[i]));
            }
        } else {
            schedule.destroy_object(Some(instance.node))?;
        }
        marker.destroy(scene);
        instance.destroyed = true;
        Ok(())
    }
    pub fn sync_positions(&mut self, scene: &crate::physics::scene::Scene) {
        for instance in &mut self.instances {
            if let State::Marker(marker) = &instance.state
                && let Some(object) = scene
                    .objects
                    .iter()
                    .find(|o| o.id == marker.object)
                    .or(marker.pending_object.as_ref())
            {
                instance.position = object.transform.position;
            }
        }
    }
    pub fn phase(
        &mut self,
        schedule: &mut scheduler::State,
        targets: &mut shockwave::Targets,
        phase: scheduler::Event,
        dt: Fix,
        space: bool,
    ) -> scheduler::Result<bool> {
        self.sync_positions(targets.scene);
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
            (State::Marker(state), Prefab::Marker(_)) => {
                if matches!(phase.kind, scheduler::Phase::Init) && id == instance.components[3] {
                    state.initialize(targets.scene, schedule.components[id].hierarchy, false)?;
                }
                if matches!(phase.kind, scheduler::Phase::Update) {
                    if id == instance.components[0] {
                        state.update(targets.scene, instance.position, space)?;
                    }
                    if id == instance.components[2] {
                        state.update_body(targets.scene)?;
                    }
                }
            }
            _ => return Err("spell_prefab_kind_mismatch"),
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
    End,
    AddScale { value: Fix },
    Position { value: Vec2 },
    ManualInit,
}
impl World {
    pub fn command(
        &mut self,
        index: usize,
        command: Command,
        schedule: &mut scheduler::State,
        targets: &mut shockwave::Targets,
    ) -> scheduler::Result<()> {
        if matches!(command, Command::End) {
            self.end_marker(index, schedule, targets.scene)?;
            self.sync_positions(targets.scene);
            return Ok(());
        }
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
                _ => return Err("shockwave_command_required"),
            },
            (State::Marker(state), Prefab::Marker(config)) => match command {
                Command::Destroy => destroy = true,
                Command::AddScale { value } => {
                    state.add_scale(targets.scene, value, config.children)?
                }
                Command::Position { value } => state.set_position(targets.scene, value)?,
                Command::ManualInit => state.initialize(
                    targets.scene,
                    schedule.components[instance.components[3]].hierarchy,
                    true,
                )?,
                _ => return Err("marker_command_required"),
            },
            _ => return Err("spell_prefab_kind_mismatch"),
        }
        if destroy {
            schedule.destroy_object(Some(instance.node))?;
            instance.destroyed = true;
            if let State::Marker(marker) = &mut instance.state {
                marker.destroy(targets.scene);
            }
        }
        self.sync_positions(targets.scene);
        Ok(())
    }
}

pub struct Spawn<'a> {
    pub prefab: &'a str,
    pub actor: usize,
    pub player: i32,
    pub position: Vec2,
    pub scale: Fix,
    pub object: Option<u32>,
}
