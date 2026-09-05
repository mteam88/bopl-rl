//! DeathEffect2: prefab registration, animation callbacks, surface queries and
//! deferred deletion. Unity's visual-only splash randomness is not gameplay RNG.
use crate::{animator, fixed::Fix, physics::scene::Scene, scheduler, vector::Vec2};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Death,
    Water,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterPrefab {
    pub frames: Vec<animator::Frame>,
    pub initial_sprite: Option<u32>,
}
pub struct Spawn {
    pub position: Vec2,
    pub actor: usize,
    pub scale: f32,
    pub material: Option<u32>,
    pub kind: Kind,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    pub anticipation: Vec<animator::Frame>,
    pub start: Vec<animator::Frame>,
    pub end: Vec<animator::Frame>,
    pub initial_sprite: Option<u32>,
    pub gibbs: i32,
    pub distance: Fix,
    pub hitstop: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Kind>,
    pub actor: usize,
    pub node: usize,
    pub components: [usize; 3],
    pub position: Vec2,
    pub scale: f32,
    pub material: Option<u32>,
    pub vibrate: bool,
    pub present: bool,
    pub destroyed: bool,
    pub animator: animator::State,
    pub splashes: Vec<Vec2>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effects {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_prefab: Option<WaterPrefab>,
    pub prefab: Prefab,
    pub instances: Vec<Effect>,
}
impl Effects {
    pub fn spawn(
        &mut self,
        schedule: &mut scheduler::State,
        position: Vec2,
        actor: usize,
        scale: f32,
        material: Option<u32>,
    ) -> scheduler::Result<()> {
        self.spawn_kind(
            schedule,
            Spawn {
                position,
                actor,
                scale,
                material,
                kind: Kind::Death,
            },
        )
    }
    pub fn spawn_kind(
        &mut self,
        schedule: &mut scheduler::State,
        request: Spawn,
    ) -> scheduler::Result<()> {
        let Spawn {
            position,
            actor,
            scale,
            material,
            kind,
        } = request;
        let water = kind == Kind::Water;
        let initial_sprite = if water {
            self.water_prefab
                .as_ref()
                .ok_or("water_effect_not_bound")?
                .initial_sprite
        } else {
            self.prefab.initial_sprite
        };
        let node = schedule.nodes.len();
        schedule.nodes.push(scheduler::Node {
            parent: None,
            children: vec![],
            active: true,
            present: true,
        });
        let first = schedule.components.len();
        let components = [first, first + 1, first + 2];
        schedule.begin_prefab();
        for (i, &id) in components.iter().enumerate() {
            schedule.components.push(scheduler::Component {
                owner: node,
                kind: if i == 2 {
                    scheduler::Kind::Fix
                } else {
                    scheduler::Kind::Updatable
                },
                present: true,
                destroyed: false,
                hierarchy: if water && i < 2 {
                    2 - i as i32
                } else {
                    (i + 1) as i32
                },
                enabled: !water || i != 0,
                ignore_activity: false,
                execution_gate: None,
            });
            if i < 2 {
                schedule.register(Some(id));
            }
        }
        schedule.end_prefab()?;
        self.instances.push(Effect {
            kind: water.then_some(kind),
            actor,
            node,
            components,
            position,
            scale,
            material,
            vibrate: false,
            present: true,
            destroyed: false,
            animator: animator::State::idle(initial_sprite),
            splashes: vec![],
        });
        Ok(())
    }
    pub fn phase(
        &mut self,
        schedule: &mut scheduler::State,
        scene: &mut Scene,
        phase: scheduler::Event,
        report: &mut crate::actors::Report,
    ) -> scheduler::Result<bool> {
        let Some(id) = phase.id else { return Ok(false) };
        if matches!(phase.kind, scheduler::Phase::DestroyObject) {
            for e in &mut self.instances {
                if e.node == id {
                    e.present = false;
                }
            }
            return Ok(false);
        }
        let Some(e) = self
            .instances
            .iter_mut()
            .find(|e| e.components.contains(&id))
        else {
            return Ok(false);
        };
        let begin = |a: &mut animator::State, frames: &Vec<animator::Frame>, callback: u32| {
            a.start(
                animator::Start::Begin {
                    frames: Some(frames.clone()),
                    next: None,
                    callback: Some(callback),
                    style: 3,
                },
                &[],
                &mut |_, _| Ok(()),
            )
        };
        if matches!(phase.kind, scheduler::Phase::Init) && id == e.components[0] {
            if e.kind == Some(Kind::Water) {
                begin(
                    &mut e.animator,
                    &self
                        .water_prefab
                        .as_ref()
                        .ok_or("water_effect_not_bound")?
                        .frames,
                    3,
                )?;
                e.animator.child_suffix = Some(String::new());
                return Ok(true);
            }
            report.events.push(crate::actors::Event {
                actor: e.actor,
                event: crate::player_controller::Event::Sound("slimeDeath".into()),
            });
            begin(&mut e.animator, &self.prefab.anticipation, 1)?;
            e.vibrate = true;
            schedule.hitstop = self.prefab.hitstop;
        } else if matches!(phase.kind, scheduler::Phase::Update) && id == e.components[1] {
            e.animator
                .update(schedule.time_scale, None, &[], &mut |animator, callback| {
                    match callback {
                        1 => {
                            begin(animator, &self.prefab.start, 2)?;
                            e.vibrate = false;
                        }
                        2 => {
                            for i in 0..self.prefab.gibbs {
                                let angle = Fix::from_int(i as i64) * Fix::ONE
                                    / Fix::from_int(self.prefab.gibbs as i64)
                                    * Fix::TAU;
                                let hit = scene.raycast_rounded(
                                    e.position,
                                    Vec2::new(angle.cos(), angle.sin()),
                                    self.prefab.distance,
                                );
                                if hit.object.is_some() {
                                    e.splashes.push(hit.near_position);
                                }
                            }
                            begin(animator, &self.prefab.end, 3)?;
                        }
                        3 => {
                            schedule.destroy_object(Some(e.node))?;
                            e.destroyed = true;
                        }
                        _ => return Err("death_effect_callback_invalid"),
                    }
                    Ok(())
                })?;
        }
        Ok(true)
    }
}
