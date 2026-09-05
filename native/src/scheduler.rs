//! Updater's registration, initialization and deferred destruction order.
//! Phase callbacks connect player/component and physics work without borrowing
//! scheduler entries across calls that can mutate the queues themselves.
use crate::{fixed::Fix, game_time, mono_sort};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
pub type Result<T> = std::result::Result<T, &'static str>;
pub type Id = usize;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pending {
    /// Mono List backing storage, including capacity. Broken comparison ordering
    /// can read beyond Count into spare null slots, so keep this observable state.
    pub items: Vec<Option<Id>>,
    pub length: usize,
}
impl Pending {
    fn add(&mut self, id: Option<Id>) {
        if self.length == self.items.len() {
            self.items.resize((self.items.len() * 2).max(4), None);
        }
        self.items[self.length] = id;
        self.length += 1;
    }
    fn clear(&mut self) {
        self.items[..self.length].fill(None);
        self.length = 0;
    }
    fn remove(&mut self, index: usize) -> Result<()> {
        if index >= self.length {
            return Err("argument_out_of_range");
        }
        self.items.copy_within(index + 1..self.length, index);
        self.length -= 1;
        self.items[self.length] = None;
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Updatable,
    Fix,
    Plain,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub parent: Option<Id>,
    pub children: Vec<Id>,
    pub active: bool,
    pub present: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    pub owner: Id,
    pub kind: Kind,
    pub present: bool,
    pub destroyed: bool,
    pub hierarchy: i32,
    pub enabled: bool,
    pub ignore_activity: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_gate: Option<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eaten_count: Option<i32>,
    pub id: i32,
    pub protected: bool,
    pub invisible: bool,
    pub time_invisible: Fix,
    pub time_ability: Fix,
    pub input: Option<Id>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flags: Option<crate::ability::PlayerFlags>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<crate::player_position::State>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<Fix>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Input {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub angle: Option<u8>,
    pub jump: bool,
    pub pressed: bool,
    pub previous: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub nodes: Vec<Node>,
    pub components: Vec<Component>,
    pub players: Vec<Player>,
    pub inputs: Vec<Input>,
    pub live: Vec<Option<Id>>,
    pub pending: Pending,
    pub prefab: Vec<Option<Id>>,
    pub destroy_buffer: Vec<Id>,
    pub destroyed_objects: VecDeque<Option<Id>>,
    pub destroyed_monos: VecDeque<Option<Id>>,
    pub added: i32,
    pub prefabs_added: i32,
    pub registering_prefab: bool,
    pub loaded: bool,
    pub stopped: bool,
    pub time_scale: Fix,
    pub time: Fix,
    pub level_time: Fix,
    pub ticks: i32,
    pub hitstop: i32,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    PlayerInputs,
    PlayerUpdate,
    Init,
    Update,
    Late,
    DestroyUpdatable,
    Duplicates,
    Magnet,
    Physics,
    RopeLate,
    RopeMesh,
    DestroyObject,
    DestroyMono,
}
impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PlayerInputs => "player_inputs",
            Self::PlayerUpdate => "player_update",
            Self::Init => "init",
            Self::Update => "update",
            Self::Late => "late",
            Self::DestroyUpdatable => "destroy_updatable",
            Self::Duplicates => "duplicates",
            Self::Magnet => "magnet",
            Self::Physics => "physics",
            Self::RopeLate => "rope_late",
            Self::RopeMesh => "rope_mesh",
            Self::DestroyObject => "destroy_object",
            Self::DestroyMono => "destroy_mono",
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Event {
    pub kind: Phase,
    pub id: Option<Id>,
}
impl Event {
    fn new(kind: Phase, id: Option<Id>) -> Self {
        Self { kind, id }
    }
}
impl State {
    pub fn register(&mut self, id: Option<Id>) {
        self.added = self.added.wrapping_add(1);
        if self.registering_prefab {
            self.prefab.push(id);
        } else {
            self.pending.add(id);
        }
    }
    pub fn begin_prefab(&mut self) {
        self.registering_prefab = true;
    }
    pub fn end_prefab(&mut self) -> Result<()> {
        self.registering_prefab = false;
        for id in &self.prefab {
            let id = id.ok_or("null_reference")?;
            let component = &mut self.components[id];
            component.hierarchy = component
                .hierarchy
                .wrapping_add(self.prefabs_added.wrapping_add(100_000_000));
            self.pending.add(Some(id));
        }
        self.prefabs_added = self.prefabs_added.wrapping_add(self.prefab.len() as i32);
        self.prefab.clear();
        Ok(())
    }
    pub fn reinit(&mut self) {
        self.time = Fix::ZERO;
        self.level_time = Fix::ZERO;
        self.ticks = 0;
        self.live.clear();
        self.prefab.clear();
        self.pending.clear();
        self.added = 0;
        self.prefabs_added = 0;
        self.registering_prefab = false;
    }
    pub fn pre_level(&mut self) {
        for p in &mut self.players {
            p.protected = false;
            p.invisible = false;
            p.time_invisible = Fix::ZERO;
            p.time_ability = Fix::ZERO;
        }
        self.time_scale = Fix::ONE;
        self.destroy_buffer.clear();
        self.live.clear();
        self.pending.clear();
        self.prefab.clear();
        self.level_time = Fix::ZERO;
    }
    fn subtree(&self, id: Id, output: &mut Vec<Id>) {
        output.push(id);
        for &child in &self.nodes[id].children {
            if self.nodes[child].present {
                self.subtree(child, output);
            }
        }
    }
    pub fn destroy_object(&mut self, id: Option<Id>) -> Result<()> {
        let id = id.ok_or("null_reference")?;
        if !self.nodes[id].present {
            return Err("null_reference");
        }
        let mut descendants = Vec::new();
        self.subtree(id, &mut descendants);
        self.destroy_buffer.clear();
        for node in descendants {
            for (i, c) in self.components.iter_mut().enumerate() {
                if c.owner == node && c.present && c.kind == Kind::Updatable {
                    self.destroy_buffer.push(i);
                    c.destroyed = true;
                }
            }
        }
        // GetComponent returns only the first fixed transform on each object;
        // only direct children are marked here, unlike updatable traversal.
        for (node, n) in self.nodes.iter().enumerate() {
            if n.present
                && (node == id || n.parent == Some(id))
                && let Some(c) = self
                    .components
                    .iter_mut()
                    .find(|c| c.present && c.owner == node && c.kind == Kind::Fix)
            {
                c.destroyed = true;
            }
        }
        self.destroyed_objects.push_back(Some(id));
        Ok(())
    }
    pub fn destroy_mono(&mut self, id: Option<Id>) {
        if let Some(id) = id
            && self.components[id].kind != Kind::Plain
        {
            self.components[id].destroyed = true;
        }
        self.destroyed_monos.push_back(id);
    }
    fn null(&self, id: Option<Id>) -> bool {
        id.is_none_or(|i| !self.components[i].present)
    }
    fn removed(&self, id: Option<Id>) -> bool {
        self.null(id) || self.components[id.unwrap()].destroyed
    }
    pub(crate) fn active(&self, id: Id) -> bool {
        let node = &self.nodes[id];
        node.present && node.active && node.parent.is_none_or(|p| self.active(p))
    }
    fn enabled(&self, id: Id) -> bool {
        let c = &self.components[id];
        c.enabled && c.execution_gate != Some(false) && (c.ignore_activity || self.active(c.owner))
    }
    pub fn sort_pending(&mut self) -> Result<()> {
        let components = &self.components;
        mono_sort::sort(&mut self.pending.items, self.pending.length, |a, b| {
            let a = a.ok_or("compare_failed")?;
            let b = b.ok_or("compare_failed")?;
            Ok(components[a]
                .hierarchy
                .wrapping_sub(components[b].hierarchy))
        })
    }
    fn cleanup(&mut self, notify: &mut impl FnMut(&mut Self, Event) -> Result<()>) -> Result<()> {
        let mut i = self.live.len() as isize - 1;
        while i >= 0 {
            let id = *self.live.get(i as usize).ok_or("argument_out_of_range")?;
            if self.removed(id) {
                // ?. uses CLR null, so a destroyed Unity object still receives
                // OnDestroyUpdatable before this list slot is removed.
                if id.is_some() {
                    notify(self, Event::new(Phase::DestroyUpdatable, id))?;
                }
                if i as usize >= self.live.len() {
                    return Err("argument_out_of_range");
                }
                self.live.remove(i as usize);
            }
            i -= 1;
        }
        Ok(())
    }
    fn finish_time(&mut self, dt: Fix) {
        self.time += dt;
        self.level_time += dt;
        self.ticks = self.ticks.wrapping_add(1);
    }
    pub fn tick(
        &mut self,
        dt: Fix,
        notify: &mut impl FnMut(&mut Self, Event) -> Result<()>,
    ) -> Result<()> {
        if self.loaded {
            self.stopped = false;
        }
        let mut i = 0;
        while i < self.players.len() {
            notify(self, Event::new(Phase::PlayerInputs, Some(i)))?;
            let input = self.players[i].input.ok_or("null_reference")?;
            let p = &mut self.inputs[input];
            p.pressed = !p.previous && p.jump;
            p.previous = p.jump;
            notify(self, Event::new(Phase::PlayerUpdate, Some(i)))?;
            let p = &mut self.players[i];
            if p.invisible {
                let dt = game_time::fixed_delta_time(dt, self.time_scale, Some(p.protected));
                p.time_invisible += dt;
                p.time_ability += dt;
                if p.time_invisible > Fix::from_int(5) {
                    p.invisible = false;
                    p.time_invisible = Fix::ZERO;
                    p.time_ability = Fix::ZERO;
                }
            } else {
                p.time_invisible = Fix::ZERO;
                p.time_ability = Fix::ZERO;
            }
            if let Some(position) = &mut p.position {
                position.begin_frame();
            }
            i += 1;
        }
        for i in (0..self.pending.length).rev() {
            if self.removed(self.pending.items[i]) {
                self.pending.remove(i)?;
            }
        }
        self.sort_pending()?;
        let mut i = 0;
        while i < self.pending.length {
            let id = self.pending.items[i];
            if !self.removed(id) {
                notify(self, Event::new(Phase::Init, id))?;
            }
            i += 1;
        }
        self.live
            .extend_from_slice(&self.pending.items[..self.pending.length]);
        self.pending.clear();
        if self.time_scale >= Fix::from_f32(0.01) {
            notify(self, Event::new(Phase::Duplicates, None))?;
        }
        self.cleanup(notify)?;
        let mut i = 0;
        while i < self.live.len() {
            let id = self.live[i];
            if !self.removed(id) && self.enabled(id.unwrap()) {
                notify(self, Event::new(Phase::Update, id))?;
            }
            i += 1;
        }
        notify(self, Event::new(Phase::Magnet, None))?;
        if self.loaded {
            self.finish_time(dt);
            self.loaded = false;
            return Ok(());
        }
        notify(self, Event::new(Phase::Physics, None))?;
        let mut i = 0;
        while i < self.live.len() {
            let id = self.live[i];
            if !self.removed(id) && self.enabled(id.unwrap()) {
                notify(self, Event::new(Phase::Late, id))?;
            }
            i += 1;
        }
        notify(self, Event::new(Phase::RopeLate, None))?;
        notify(self, Event::new(Phase::RopeMesh, None))?;
        self.cleanup(notify)?;
        while let Some(id) = self.destroyed_objects.pop_front() {
            notify(self, Event::new(Phase::DestroyObject, id))?;
            if let Some(id) = id {
                let mut nodes = Vec::new();
                self.subtree(id, &mut nodes);
                for node in nodes {
                    self.nodes[node].present = false;
                    for c in &mut self.components {
                        if c.owner == node {
                            c.present = false;
                        }
                    }
                }
            }
        }
        while let Some(id) = self.destroyed_monos.pop_front() {
            notify(self, Event::new(Phase::DestroyMono, id))?;
            if let Some(id) = id {
                self.components[id].present = false;
            }
        }
        self.finish_time(dt);
        Ok(())
    }
}
