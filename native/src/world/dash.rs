//! Dash body switching and exact per-tick ability timing.
use super::*;

impl Player {
    pub(super) fn enter_dash(&mut self, action: Action, movement_input: Vec2, terrain: Terrain) {
        let platforms = terrain.platforms;
        let mut old = self.clone();
        // OldUpdate enters the ability, but SlimeController.UpdateSim still calls
        // Move once on its now-inactive original body before returning.
        if old.grounded {
            old.move_grounded(movement_input, terrain);
        } else {
            old.velocity += Vec2::new(movement_input.x * old.constants.air_accel, Fix::ZERO);
            old.velocity += (-Vec2::RIGHT)
                * (old.constants.air_accel / (old.constants.max_speed + old.constants.air_accel))
                * old.velocity.x;
            old.collision_rays(terrain);
        }
        self.dash.saved_local = old.local;
        if self.grounded {
            self.dash.local = self.local;
            self.position = self
                .platform(platforms)
                .unwrap()
                .surface
                .position(self.local, self.dash.radius);
            self.position += self
                .platform(platforms)
                .unwrap()
                .surface
                .normal(self.position)
                * Fix::from_f32(0.05);
        } else {
            self.ground = None;
        }
        self.local = self.dash.local;
        self.grounded = false;
        self.grounded_speed = Fix::ZERO;
        self.rotation = Fix::ZERO;
        self.up = Vec2::UP;
        self.velocity = Vec2::ZERO;
        self.external_velocity = Vec2::ZERO;
        self.dash.active = true;
        self.dash.time = Fix::ZERO;
        self.dash.updates = 0;
        self.dash.in_dash = false;
        self.dash.input = action.direction();
        self.cooldown = Fix::ZERO;
    }
    pub(super) fn dash_step(&mut self, action: Action, terrain: Terrain) {
        let platforms = terrain.platforms;
        self.dash.time += DT;
        if self.dash.time < Fix::from_raw(42_949_672) {
            let aim = action.direction();
            if aim.length() > Fix::from_f32(0.4) {
                self.dash.input = aim;
            }
        } else if !self.dash.in_dash {
            self.dash.in_dash = true;
            let mut input = self.dash.input;
            if input.length() < Fix::from_f32(0.4) {
                input = Vec2::UP;
            }
            self.dash.input = input.normalized();
            let angle = self
                .dash
                .input
                .normalized()
                .dot(Vec2::UP)
                .acos()
                .expect("normalized direction");
            self.rotation = if self.dash.input.x > Fix::ZERO {
                Fix::TAU - angle
            } else {
                angle
            };
            self.up = Vec2::angle(self.rotation).perp();
            self.dash.updates = 0;
            self.velocity = Vec2::ZERO;
            self.external_velocity = Vec2::ZERO;
            if let Some((index, _, position)) = closest(
                terrain,
                self.position,
                self.dash.input,
                Fix::from_int(13),
                None,
            ) {
                let platform = &platforms[index];
                self.dash.hit_ground = Some(platform.id);
                self.dash.hit_local = platform
                    .surface
                    .local_position(
                        platform.surface.closest_point(position).unwrap(),
                        self.dash.radius,
                    )
                    .unwrap();
            } else {
                self.dash.hit_ground = None;
            }
        } else {
            if self.dash.updates > 2 {
                // Unity's `hitGround != null` is false once the platform has been
                // destroyed mid-dash, so the C# falls through to the free dash.
                let landing = self
                    .dash
                    .hit_ground
                    .and_then(|id| platforms.iter().find(|p| p.id == id));
                if let Some(platform) = landing {
                    self.position =
                        platform.surface.position(self.dash.hit_local, self.dash.radius);
                    // Dash's own PlayerPhysics attaches, with its radius.
                    let radius = std::mem::replace(&mut self.radius, self.dash.radius);
                    self.attach(platform, self.position);
                    self.radius = radius;
                    self.dash.local = self.local;
                } else {
                    let mut position = self.position + self.dash.input * Fix::from_int(13);
                    let water = Fix::from_f32(-11.3);
                    if !self.space && position.y < water {
                        let distance = (water - self.position.y) / self.dash.input.y;
                        position.y = water;
                        position.x = self.position.x + self.dash.input.x * distance;
                    }
                    self.position = position;
                }
                self.dash.exit_rotation = self.rotation;
                // Ability.ExitAbility publishes `potentialCollider.position`
                // before `onExitAbility` restores the slime, so this raw
                // endpoint - not the ray-corrected body below - is what
                // Player.Position carries on the exit pass.
                self.dash.exit_position = self.position;
                self.dash.active = false;
                self.ground = None;
                self.grounded = false;
                self.grounded_speed = Fix::ZERO;
                self.rotation = Fix::ZERO;
                self.up = Vec2::UP;
                self.local = self.dash.saved_local;
                self.velocity = self.dash.input
                    * if self.space {
                        Fix::from_int(30)
                    } else {
                        Fix::from_int(60)
                    };
                self.external_velocity = Vec2::ZERO;
                if self.dash.exit_radius != Fix::ZERO {
                    self.radius = self.dash.exit_radius;
                }
                // SlimeController.onExitAbility immediately runs the returned
                // slime's velocity rays with Dash's leave velocity. Keep this
                // physical handoff here so both the scalar World and batched
                // rollout paths can land on the ejection tick.
                self.collision_rays(terrain);
            }
            self.dash.updates += 1;
        }
        if self.dash.active && self.outside_bounds() {
            self.alive = false;
        }
    }
}
