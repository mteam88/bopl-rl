//! Revival ability marker and physical component state.
use crate::{
    actor_collider::{CircleComponent, Component},
    bopl_body,
    fixed::Fix,
    physics::{
        self,
        scene::{Object, Scene, Transform},
        shape::Collider,
    },
    scheduler,
    vector::Vec2,
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub body: bopl_body::Config,
    pub circle: CircleComponent,
    pub layer: usize,
    pub offset: Vec2,
    pub particle: bool,
    pub children: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Marker {
    pub owner: i32,
    pub object: u32,
    pub body: bopl_body::State,
    pub circle: Component,
    pub scale: Fix,
    pub pending_object: Option<Object>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ability {
    pub player: Option<i32>,
    pub indicator: Option<usize>,
}
pub fn camera_clamp(mut position: Vec2) -> Vec2 {
    position.x = position
        .x
        .max(Fix::from_f32(-97.27) + Fix::from_int(2))
        .min(Fix::from_f32(97.6) - Fix::from_int(2));
    position.y = position.y.max(Fix::from_int(-24)).min(Fix::from_int(38));
    position
}
impl Marker {
    pub fn new(config: &Config, object: u32, owner: i32, position: Vec2) -> Self {
        Self {
            owner,
            object,
            scale: Fix::ONE,
            body: bopl_body::State {
                config: config.body.clone(),
                resizable: false,
                prev_scale: Fix::ONE,
                frames_until_drop: -1,
                magnet: None,
                sorting: None,
                original_sorting: bopl_body::Sorting {
                    layer: None,
                    order: 0,
                },
            },
            circle: Component::Circle {
                circle: config.circle.clone(),
            },
            pending_object: Some(Object {
                id: object,
                hierarchy: Some(3),
                collider: Collider::Circle(crate::geometry::Circle {
                    center: position + config.offset,
                    radius: config.circle.start_radius,
                }),
                body: physics::Body::default(),
                transform: Transform {
                    position,
                    rotation: Fix::ZERO,
                    offset: config.offset,
                    right: None,
                },
                layer: config.layer,
                active: true,
                enabled: true,
                callback: true,
                callback_destroyed: false,
                destroyed: false,
                composite: None,
                relative_pos: Vec2::ZERO,
                relative_orientation: Fix::ZERO,
            }),
        }
    }
    pub fn initialize(
        &mut self,
        scene: &mut Scene,
        hierarchy: i32,
        manual: bool,
    ) -> scheduler::Result<()> {
        if !manual && self.circle.skips_init() {
            return Ok(());
        }
        let object = self
            .pending_object
            .as_mut()
            .ok_or("spell_collider_already_initialized")?;
        self.circle.initialize_geometry(object)?;
        object.body = self.body.config.new_body(object.transform.rotation)?;
        object.hierarchy = Some(hierarchy);
        scene.objects.push(self.pending_object.take().unwrap());
        Ok(())
    }
    pub fn update_body(&mut self, scene: &mut Scene) -> scheduler::Result<()> {
        self.body.update(&mut bopl_body::Context {
            scene,
            object: self.object,
            initialized: self.circle.initialized(),
            transform_scale: self.scale,
        })
    }
    pub fn set_position(&mut self, scene: &mut Scene, position: Vec2) -> scheduler::Result<()> {
        scene
            .objects
            .iter_mut()
            .find(|o| o.id == self.object)
            .or(self.pending_object.as_mut())
            .ok_or("missing_spell_object")?
            .transform
            .position = position;
        Ok(())
    }
    pub fn update(
        &mut self,
        scene: &mut Scene,
        position: Vec2,
        space: bool,
    ) -> scheduler::Result<()> {
        let mut position = camera_clamp(position);
        if !space && position.y < Fix::from_f32(-11.3) {
            position.y = Fix::from_f32(-11.3);
        }
        self.set_position(scene, position)
    }
    pub fn add_scale(
        &mut self,
        scene: &mut Scene,
        diff: Fix,
        children: usize,
    ) -> scheduler::Result<()> {
        let Component::Circle { circle } = &self.circle else {
            return Err("marker_circle_required");
        };
        let value = circle.scale + diff;
        self.circle.set_scale(
            value,
            &mut self.scale,
            self.object,
            scene,
            self.pending_object.as_mut(),
        )?;
        if children < 2 {
            return Err("index_out_of_range");
        }
        Ok(())
    }
    pub fn destroy(&mut self, scene: &mut Scene) {
        if let Some(object) = scene
            .objects
            .iter_mut()
            .find(|o| o.id == self.object)
            .or(self.pending_object.as_mut())
        {
            object.destroyed = true;
        }
    }
}

impl Ability {
    pub fn set_flag(
        &mut self,
        marker: Option<usize>,
        player: i32,
        actor: usize,
        context: &mut crate::actors::Context,
    ) -> scheduler::Result<()> {
        let world = context.spells.as_mut().ok_or("spell_world_not_bound")?;
        if let Some(previous) = self
            .indicator
            .filter(|&i| world.instances[i].present && !world.instances[i].destroyed)
        {
            world.end_marker(previous, context.schedule, context.scene)?;
            let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
            let index = combat
                .book
                .players
                .iter()
                .position(|p| Some(p.id) == self.player)
                .ok_or("null_reference")?;
            let positions = combat.revival[index]
                .positions
                .as_mut()
                .ok_or("missing_revival_positions")?;
            let mut i = positions.len() as isize - 1;
            while i >= 0 && (i as usize) < positions.len() {
                if positions[i as usize]
                    .is_none_or(|j| !world.instances[j].present || world.instances[j].destroyed)
                {
                    positions.remove(i as usize);
                } else {
                    i -= 1;
                }
            }
            combat.revival[index].respawn_count = positions.len();
        }
        let marker = marker.ok_or("null_reference")?;
        self.indicator = matches!(
            world.instances[marker].state,
            crate::spell_world::State::Marker(_)
        )
        .then_some(marker);
        self.player = context
            .schedule
            .players
            .iter()
            .find(|p| p.id == player)
            .map(|p| p.id);
        let combat = context.combat.as_mut().ok_or("missing_actor_combat")?;
        let index = combat
            .book
            .players
            .iter()
            .position(|p| Some(p.id) == self.player)
            .ok_or("null_reference")?;
        let revival = &mut combat.revival[index];
        revival.status = 1;
        revival.ability = Some(actor);
        let positions = revival
            .positions
            .as_mut()
            .ok_or("missing_revival_positions")?;
        positions.push(self.indicator);
        revival.respawn_count = positions.len();
        Ok(())
    }
}
