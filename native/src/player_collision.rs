//! PlayerCollision's lethality decisions. Effects run in source order through
//! the caller; death lifecycle and ground attachment are separate handlers.
use crate::{fixed::Fix, physics::shape::Collider, player_life::Cause, vector::Vec2};
use serde::{Deserialize, Serialize};

// Pinned TagManager layer indices. The source resolves these names at each call.
const PLAYER: i32 = 8;
const PROJECTILE: i32 = 9;
const WALL: i32 = 11;
const LETHAL_TERRAIN: i32 = 16;
const EXPLOSION: i32 = 18;
const ORB: i32 = 26;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ignored {
    pub hierarchy: i32,
    pub destroyed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attached {
    pub body: u32,
    pub destroyed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ground {
    pub grounded: bool,
    pub attached: Option<Attached>,
}
/// Resolved component state at a contact. Live collider references (including
/// those copied by CopyIgnoredColliderFrom) must be resolved again each contact;
/// these metadata snapshots do not own entity identity or destruction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receiver {
    pub player_id: i32,
    pub player_alive: Option<bool>,
    pub invulnerable: u32,
    pub was_killed: bool,
    pub position: Vec2,
    pub right: Vec2,
    pub squish_offset: Vec2,
    pub squishable: bool,
    pub physics: Option<Ground>,
    pub ignored: [Option<Ignored>; 6],
    pub ignored_beam: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Engine {
    pub owner: i32,
    pub destroyed: bool,
    pub on: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Platform {
    pub body: u32,
    pub pickup: bool,
    pub engine: Option<Engine>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Explosion {
    pub owner: i32,
    pub hurt_owner: bool,
    pub beam: bool,
    pub beam_id: Option<i32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Projectile {
    pub owner: i32,
    pub hurt_owner: bool,
    pub age: Fix,
    pub arrow_destroyed: Option<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rock {
    pub hits: i32,
    pub player_local: Option<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ability {
    pub owner: i32,
    pub kills_on_contact: bool,
    pub rock: Option<Rock>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub id: u32,
    pub layer: i32,
    pub collision_layer: i32,
    pub hierarchy: i32,
    pub transform_present: bool,
    pub destroyed: bool,
    pub collider: Collider,
    pub platform: Option<Platform>,
    pub orb_kind: Option<i32>,
    pub explosion: Option<Explosion>,
    pub hitbox_owner: Option<i32>,
    pub projectile: Option<Projectile>,
    pub holder_owner: Option<i32>,
    pub ability: Option<Ability>,
    pub parent_ability: Option<Ability>,
    pub chain_lightning: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Globals {
    pub players: Vec<i32>,
    pub local_players: Vec<i32>,
    pub replay: bool,
    pub engine_achievement: bool,
    pub arrow_achievement: bool,
    pub rock_achievement: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    Attach { target: u32, position: Vec2 },
    Lightning,
    Kill { killer: i32, cause: i32 },
}

impl Receiver {
    /// UpdateSim clears only the per-frame duplicate-kill guard.
    pub fn update(&mut self) {
        self.was_killed = false;
    }
    pub fn set_invulnerable(&mut self, enabled: bool, mask: u32) {
        if enabled {
            self.invulnerable |= mask;
        } else {
            self.invulnerable &= !mask;
        }
    }
    /// `callback` selects OnCollide (guarded) or direct KillPlayerOnCollision.
    /// Kill requests use spawnEffect=true and ignoreInvulnerability=false.
    /// The handler return value is the direct collision result; OnCollide ignores it.
    pub fn collide(
        &mut self,
        hit: &mut Hit,
        globals: &mut Globals,
        callback: bool,
        effect: &mut impl FnMut(Effect) -> bool,
    ) -> Result<bool, &'static str> {
        if (callback && self.was_killed)
            || self.invulnerable != 0
            || self.player_alive == Some(false)
            || !hit.transform_present
            || hit.destroyed
        {
            return Ok(false);
        }
        if self
            .ignored
            .iter()
            .flatten()
            .any(|c| !c.destroyed && c.hierarchy == hit.hierarchy && hit.hierarchy != 0)
        {
            return Ok(false);
        }
        let mut squished = false;
        if hit.collision_layer == WALL
            && let (true, Some(platform), Some(ground)) =
                (self.squishable, &hit.platform, &self.physics)
            && ground.grounded
            && ground
                .attached
                .as_ref()
                .is_some_and(|a| !a.destroyed && a.body != platform.body)
        {
            if platform.pickup {
                effect(Effect::Attach {
                    target: hit.id,
                    position: self.position,
                });
            } else {
                let point = self.position + self.right.complex_mul(self.squish_offset);
                let overlap = match hit.collider {
                    Collider::Circle(c) => c.overlaps(point),
                    Collider::Box(b) => b.overlaps(point),
                    Collider::RoundedRect(r) => r.overlaps(point),
                };
                if overlap {
                    if !globals.engine_achievement
                        && let Some(engine) = &platform.engine
                        && !engine.destroyed
                        && engine.on
                    {
                        if self.player_alive.is_none() {
                            return Err("null_reference");
                        }
                        if engine.owner != self.player_id {
                            // The source dereferences a missing engine owner.
                            if !globals.players.contains(&engine.owner) {
                                return Err("null_reference");
                            }
                            if globals.local_players.contains(&engine.owner) && !globals.replay {
                                globals.engine_achievement = true;
                            }
                        }
                    }
                    squished = true;
                }
            }
        }
        if !matches!(
            hit.layer,
            PROJECTILE | EXPLOSION | PLAYER | LETHAL_TERRAIN | ORB
        ) && !squished
        {
            return Ok(false);
        }
        if hit.layer == ORB && hit.orb_kind.ok_or("null_reference")? != 0 {
            return Ok(false);
        }
        let mut beam = false;
        let (killer, hurt_owner) = match hit.layer {
            EXPLOSION | LETHAL_TERRAIN => {
                let pair = if let Some(explosion) = &hit.explosion {
                    beam = explosion.beam;
                    if self.ignored_beam >= 0
                        && beam
                        && let Some(id) = explosion.beam_id
                        && id.checked_abs().ok_or("overflow")? == self.ignored_beam
                    {
                        return Ok(false);
                    }
                    (explosion.owner, explosion.hurt_owner)
                } else {
                    (hit.hitbox_owner.ok_or("null_reference")?, true)
                };
                if self.player_alive.is_some()
                    && hit.chain_lightning
                    && (pair.0 != self.player_id || pair.1)
                {
                    effect(Effect::Lightning);
                }
                pair
            }
            PROJECTILE | ORB => {
                if let Some(projectile) = &hit.projectile {
                    if !globals.arrow_achievement
                        && projectile.owner != self.player_id
                        && projectile.age >= Fix::from_f64(2.5)
                        && globals.local_players.contains(&projectile.owner)
                        && projectile.arrow_destroyed == Some(false)
                        && !globals.replay
                    {
                        globals.arrow_achievement = true;
                    }
                    (projectile.owner, projectile.hurt_owner)
                } else {
                    (hit.holder_owner.ok_or("null_reference")?, false)
                }
            }
            PLAYER => {
                if let Some(ability) = &mut hit.ability {
                    if !globals.rock_achievement
                        && ability.kills_on_contact
                        && let Some(rock) = &mut ability.rock
                    {
                        rock.hits = rock.hits.wrapping_add(1);
                        if rock.hits >= 2 && rock.player_local == Some(true) && !globals.replay {
                            globals.rock_achievement = true;
                        }
                    }
                    if !ability.kills_on_contact {
                        return Ok(false);
                    }
                    (ability.owner, true)
                } else {
                    let Some(ability) = &hit.parent_ability else {
                        return Ok(false);
                    };
                    if !ability.kills_on_contact {
                        return Ok(false);
                    }
                    (ability.owner, true)
                }
            }
            _ => (1000, false),
        };
        if killer == self.player_id && !hurt_owner {
            return Ok(false);
        }
        self.was_killed = true;
        let cause = if squished {
            Cause::Squished
        } else if killer == self.player_id && hurt_owner {
            Cause::Accident
        } else if beam {
            Cause::Beamed
        } else {
            Cause::Other
        };
        Ok(effect(Effect::Kill {
            killer,
            cause: cause as i32,
        }))
    }
}

/// SlimeController.OnCollide's overlap shove direction, after the player-layer
/// and bite checks. Priority subtraction is performed with C# i32 wrapping by
/// the caller. Physics must establish an actual collider contact first.
pub fn overlap_push_direction(
    position: Vec2,
    up: Vec2,
    radius: Fix,
    other: Vec2,
    priority_difference: i32,
) -> Option<Vec2> {
    let offset = position - other;
    let distance = offset.length();
    if distance != Fix::ZERO && distance < radius {
        Some(offset / distance)
    } else if distance == Fix::ZERO {
        let forward = up.perp() / Fix::from_int(10);
        Some(if priority_difference < 0 {
            forward
        } else {
            -forward
        })
    } else {
        None
    }
}
