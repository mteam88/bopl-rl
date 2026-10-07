//! Dash body switching and exact per-tick ability timing.
use super::*;

impl Player {
    pub(super) fn enter_dash(
        &mut self,
        action: Action,
        movement_input: Vec2,
        walls: &mut impl Walls,
        inverse_extents: Vec2,
    ) {
        let mut old = self.clone();
        // OldUpdate enters the ability, but SlimeController.UpdateSim still calls
        // Move once on its now-inactive original body before returning.
        if old.grounded {
            old.move_grounded(movement_input, walls);
        } else {
            old.velocity += Vec2::new(movement_input.x * old.constants.air_accel, Fix::ZERO);
            old.velocity += (-Vec2::RIGHT)
                * (old.constants.air_accel / (old.constants.max_speed + old.constants.air_accel))
                * old.velocity.x;
            old.collision_rays(walls);
        }
        self.dash.saved_local = old.local;
        // Dash.OnEnterAbility syncs its hitbox to the slime's pose before
        // readyDash ungrounds the body; DetPhysics moves it onto the body only
        // at its next sync.
        self.dash.active = true;
        self.sync_collider(inverse_extents);
        // readyDash's UnGround (Dash.cs:109). SyncPhysicsTo and UnGround read
        // the retained attachedGround, which a Blink may have deactivated.
        if self.grounded {
            self.dash.local = self.local;
            let surface = &self.attached_ground(walls.platforms()).unwrap().surface;
            self.position = surface.position(self.local, self.dash.radius);
            self.unground(Some(surface), true);
        } else {
            self.ground = None;
            self.unground(None, true);
        }
        self.local = self.dash.local;
        self.velocity = Vec2::ZERO;
        self.external_velocity = Vec2::ZERO;
        self.dash.time = Fix::ZERO;
        self.dash.updates = 0;
        self.dash.in_dash = false;
        self.dash.input = action.direction();
        self.cooldown = Fix::ZERO;
    }
    pub(super) fn dash_step(&mut self, action: Action, walls: &mut impl Walls) {
        // timePassed += GameTime.FixedDeltaTime (Dash.cs:246): the
        // anticipation holds under Time Stop, while updatesPassedInDash
        // below counts raw passes, so a dash already under way lands.
        self.dash.time += self.fixed_delta_time;
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
            self.set_rotation(if self.dash.input.x > Fix::ZERO {
                Fix::TAU - angle
            } else {
                angle
            });
            self.dash.updates = 0;
            self.velocity = Vec2::ZERO;
            self.external_velocity = Vec2::ZERO;
            if let Some((index, _, position)) =
                walls.raycast_to_closest_rounded_rect(self.position, self.dash.input, Fix::from_int(13), None)
            {
                let platform = &walls.platforms()[index];
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
                    .and_then(|id| walls.platforms().iter().find(|p| p.id == id));
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
                    // SceneBounds.WaterHeight, -50 on a space level.
                    let water =
                        if self.space { Fix::from_int(-50) } else { Fix::from_f32(-11.3) };
                    if position.y < water {
                        let distance = (water - self.position.y) / self.dash.input.y;
                        position.y = water;
                        position.x = self.position.x + self.dash.input.x * distance;
                    }
                    self.position = position;
                }
                self.dash.exit_rotation = self.rotation();
                // The endpoint the rope exit pins; the rollout's Dash exit
                // hands the slime over from here.
                self.dash.exit_position = self.position;
                self.dash.active = false;
            }
            self.dash.updates += 1;
        }
        if self.dash.active && self.outside_bounds() {
            self.alive = false;
            self.out_of_bounds = true;
        }
    }
}
