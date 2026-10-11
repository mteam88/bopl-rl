//! Game events (docs/STYLES.md part 1): the deaths and Blink hits a command
//! produced, in the order the game produced them, read through
//! `bopl_rl_events`. Recording only reads the simulation: nothing here feeds
//! back into it.
//!
//! Entity ids are recorded as the sim's internal ids while the command runs
//! and translated to published ids (`observed_ids`) after its last
//! observation, so events never number an entity themselves. An entity the
//! observation never listed publishes 0.
use super::*;

/// `BoplRlEvent` (sim/abi/bopl_rl.h).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct CEvent {
    pub kind: u32,
    pub tick: u32,
    pub subject: u32,
    pub actor: u32,
    pub entity: u32,
    pub source_ability: u32,
    pub cause: u32,
    pub flags: u32,
    pub related: u32,
    pub related_actor: u32,
    pub related_tick: u32,
    pub reserved: u32,
    pub x: i64,
    pub y: i64,
    pub vx: i64,
    pub vy: i64,
    pub ax: i64,
    pub ay: i64,
    pub avx: i64,
    pub avy: i64,
}

const _: () = assert!(std::mem::size_of::<CEvent>() == 112 && std::mem::align_of::<CEvent>() == 8);

pub const EVENT_DEATH: u32 = 1;
pub const EVENT_BLINK: u32 = 2;

pub const DEATH_UNKNOWN: u32 = 0;
pub const DEATH_CONTACT: u32 = 1;
pub const DEATH_PROJECTILE: u32 = 2;
pub const DEATH_EXPLOSION: u32 = 3;
pub const DEATH_BEAM: u32 = 4;
pub const DEATH_SQUISHED: u32 = 5;
pub const DEATH_EATEN: u32 = 6;
pub const DEATH_RING_OUT: u32 = 7;
pub const DEATH_SELF: u32 = 8;
pub const DEATH_OTHER: u32 = 9;

pub const DEATH_ACTOR_IN_FORM: u32 = 1;
pub const DEATH_LAST_BODY: u32 = 2;
pub const DEATH_RELATED_BLINKED: u32 = 4;

/// BLINK flags: bit 0 is the contract's "a player"; the rest refine it.
pub const BLINK_PLAYER: u32 = 1;
pub const BLINK_PLATFORM: u32 = 2;
pub const BLINK_CLONE: u32 = 4;
pub const BLINK_SELF: u32 = 8;

/// A death's `related` entity counts as blinked while its tunnel holds it
/// and for this many ticks after.
pub const BLINK_WINDOW: u32 = 60;

/// A slime body: a player slot (`players`) or an abilityless clone.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) enum BodyRef {
    #[default]
    None,
    Body(usize),
    Clone(usize),
}

/// What the kill call site knows about the death it is about to cause. Set
/// right before `kill_with`/`kill_abilityless_clone`, which take it at
/// entry, so a nested kill never sees its caller's.
#[derive(Clone, Copy, Default)]
pub(super) struct KillSource {
    /// Entity table index of the killing collider.
    pub entity: Option<usize>,
    /// The body that did it (a form's, a clone's, an eater's).
    pub actor: BodyRef,
    /// A DEATH_* the site knows, DEATH_UNKNOWN to derive it.
    pub cause: u32,
    /// `actor` killed with its own form body.
    pub in_form: bool,
    /// Source ability when neither the form nor the entity row names one.
    pub ability: u32,
    /// The victim as it was when the outer kill began (a clone promoted to
    /// an extra body for its revival).
    pub victim: Option<Victim>,
}

impl KillSource {
    pub fn ring_out() -> Self {
        Self { cause: DEATH_RING_OUT, ..Self::default() }
    }

    pub fn entity(entity: Option<usize>, cause: u32) -> Self {
        Self { entity, cause, ..Self::default() }
    }
}

/// The last ground a body stood on: the platform (id + 1, 0 none), its
/// internal entity id and the header tick it was last seen there.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(super) struct GroundMark {
    platform: u32,
    entity_id: u32,
    tick: u32,
}

#[derive(Clone, Copy, Default)]
pub(super) struct Victim {
    position: Vec2,
    velocity: Vec2,
    ground: GroundMark,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlinkTarget {
    /// By platform id: a removal reindexes `platforms`.
    Platform(u32),
    Body(usize),
    Clone(usize),
    Object(usize),
}

#[derive(Clone, Copy, Debug)]
struct BlinkMark {
    entity_id: u32,
    target: BlinkTarget,
    blinker: u32,
    last_hidden: u32,
}

#[derive(Clone)]
pub struct EventLog {
    pub out: Vec<CEvent>,
    pub(super) source: KillSource,
    ground: [GroundMark; MAX_BODIES],
    clone_ground: Vec<GroundMark>,
    blinks: Vec<BlinkMark>,
}

impl Default for EventLog {
    fn default() -> Self {
        Self {
            out: Vec::with_capacity(16),
            source: KillSource::default(),
            ground: [GroundMark::default(); MAX_BODIES],
            clone_ground: Vec::with_capacity(MAX_CLONES as usize),
            blinks: Vec::with_capacity(8),
        }
    }
}

impl EventLog {
    /// A new episode: no events, no history.
    pub fn reset(&mut self) {
        self.out.clear();
        self.source = KillSource::default();
        self.ground = [GroundMark::default(); MAX_BODIES];
        self.clone_ground.clear();
        self.blinks.clear();
    }

    /// A fork keeps the history that later events read, not the last
    /// command's events.
    pub fn fork_from(&mut self, src: &EventLog) {
        self.out.clear();
        self.source = KillSource::default();
        self.ground = src.ground;
        self.clone_ground.clone_from(&src.clone_ground);
        self.blinks.clone_from(&src.blinks);
    }
}

fn pose(position: Vec2, velocity: Vec2) -> [i64; 4] {
    Fix::raw_array([position.x, position.y, velocity.x, velocity.y])
}

impl RolloutWorld {
    /// The header tick of the first observation that shows what happens now:
    /// `game_tick` bumps `tick` at its end.
    #[inline]
    fn event_tick(&self) -> u32 {
        self.tick.wrapping_add(1)
    }

    fn ground_entity_id(&self, platform_id: u32) -> u32 {
        self.platforms
            .iter()
            .position(|platform| platform.id == platform_id)
            .and_then(|p| self.entities.get(self.platform_entities[p]))
            .map_or(0, |row| row.id)
    }

    fn body_base(&self, body: BodyRef) -> Option<&Player> {
        match body {
            BodyRef::None => None,
            BodyRef::Body(i) => self.players.get(i).map(|slot| &slot.base),
            BodyRef::Clone(i) => self.abilityless_clones.get(i).map(|clone| &clone.base),
        }
    }

    /// Once per game tick, before `tick` advances: each living body's ground,
    /// and how long each blinked entity has been out.
    pub(super) fn track_event_history(&mut self) {
        let now = self.event_tick();
        for index in 0..self.players.len().min(MAX_BODIES) {
            let base = &self.players[index].base;
            let (alive, ground) = (base.alive, base.ground);
            if !alive {
                self.events.ground[index] = GroundMark::default();
            } else if let Some(id) = ground {
                if self.events.ground[index].platform != id.wrapping_add(1) {
                    let entity_id = self.ground_entity_id(id);
                    self.events.ground[index] = GroundMark { platform: id.wrapping_add(1), entity_id, tick: now };
                } else {
                    self.events.ground[index].tick = now;
                }
            }
        }
        if self.abilityless_clones.len() != self.events.clone_ground.len() {
            self.events.clone_ground.resize(self.abilityless_clones.len(), GroundMark::default());
        }
        for index in 0..self.abilityless_clones.len() {
            let clone = &self.abilityless_clones[index];
            let (alive, ground) = (clone.active && clone.base.alive, clone.base.ground);
            if !alive {
                self.events.clone_ground[index] = GroundMark::default();
            } else if let Some(id) = ground {
                if self.events.clone_ground[index].platform != id.wrapping_add(1) {
                    let entity_id = self.ground_entity_id(id);
                    self.events.clone_ground[index] = GroundMark { platform: id.wrapping_add(1), entity_id, tick: now };
                } else {
                    self.events.clone_ground[index].tick = now;
                }
            }
        }
        if !self.events.blinks.is_empty() {
            let mut k = 0;
            while k < self.events.blinks.len() {
                let mark = self.events.blinks[k];
                if self.blink_target_hidden(mark.target, mark.entity_id) {
                    self.events.blinks[k].last_hidden = now;
                } else if now.wrapping_sub(mark.last_hidden) > BLINK_WINDOW {
                    self.events.blinks.swap_remove(k);
                    continue;
                }
                k += 1;
            }
        }
    }

    fn blink_target_hidden(&self, target: BlinkTarget, entity_id: u32) -> bool {
        let row_is = |entity: usize| self.entities.get(entity).is_some_and(|row| row.id == entity_id);
        match target {
            BlinkTarget::Platform(id) => self
                .platforms
                .iter()
                .position(|platform| platform.id == id)
                .is_some_and(|p| self.platform_dynamics[p].blink_tunnel_active),
            BlinkTarget::Body(i) => self.players.get(i).is_some_and(|slot| {
                slot.blink_tunnel_active && row_is(Self::hurtbox_entity(slot))
            }),
            BlinkTarget::Clone(i) => self.abilityless_clones.get(i).is_some_and(|clone| {
                clone.blink_tunnel_active && clone.entity.is_some_and(row_is)
            }),
            BlinkTarget::Object(entity) => row_is(entity) && self.blink_object_hidden(Some(entity)),
        }
    }

    /// The victim's pose and last ground, read as its kill begins.
    fn victim_snapshot(&self, body: BodyRef) -> Victim {
        let Some(base) = self.body_base(body) else {
            return Victim::default();
        };
        let ground = match base.ground {
            Some(id) => GroundMark { platform: id.wrapping_add(1), entity_id: self.ground_entity_id(id), tick: self.event_tick() },
            None => match body {
                BodyRef::Body(i) => self.events.ground.get(i).copied().unwrap_or_default(),
                BodyRef::Clone(i) => self.events.clone_ground.get(i).copied().unwrap_or_default(),
                BodyRef::None => GroundMark::default(),
            },
        };
        Victim { position: base.position, velocity: base.velocity + base.external_velocity, ground }
    }

    /// The death event of a kill that begins now, from the call site's
    /// source (taken) and the victim as it stands, and that source with the
    /// victim filled in, for a kill this one hands on. `player_kill`
    /// finishes and records the event.
    pub(super) fn begin_death(&mut self, victim: BodyRef, killer: u32, game_cause: u32) -> (CEvent, KillSource) {
        let mut source = std::mem::take(&mut self.events.source);
        let snapshot = *source.victim.get_or_insert_with(|| self.victim_snapshot(victim));
        let owner = match victim {
            BodyRef::Body(i) => self.players[i].owner,
            BodyRef::Clone(i) => self.abilityless_clones[i].owner,
            BodyRef::None => 0,
        };
        let subject = self.players[owner].base.id;
        // The credited killer, as Player.Kill credits `kills`.
        let actor = if killer > 0 && self.players.iter().any(|slot| slot.base.id == killer) { killer } else { 0 };
        let cause = if source.cause == DEATH_RING_OUT {
            DEATH_RING_OUT
        } else if game_cause == CAUSE_SQUISHED {
            DEATH_SQUISHED
        } else if game_cause == CAUSE_EATEN {
            DEATH_EATEN
        } else if killer == subject {
            DEATH_SELF
        } else if source.cause != DEATH_UNKNOWN {
            source.cause
        } else if game_cause == CAUSE_BEAMED {
            DEATH_BEAM
        } else if killer == ENVIRONMENT || actor != 0 {
            DEATH_OTHER
        } else {
            DEATH_UNKNOWN
        };
        let mut flags = 0;
        let mut source_ability = ABILITY_NULL;
        let actor_form = match source.actor {
            BodyRef::Body(i) if source.in_form => {
                let slot = &self.players[i];
                (slot.active != ActiveAbility::None).then(|| slot.abilities[slot.active_slot.min(2)])
            }
            _ => None,
        };
        if let Some(ability) = actor_form {
            flags |= DEATH_ACTOR_IN_FORM;
            source_ability = ability;
        }
        let entity_row = source.entity.and_then(|entity| self.entities.get(entity));
        if source_ability == ABILITY_NULL {
            source_ability = entity_row.map_or(ABILITY_NULL, |row| row.reserved[0]);
        }
        if source_ability == ABILITY_NULL {
            source_ability = source.ability;
        }
        let actor_pose = match self.body_base(source.actor) {
            Some(base) => Some(pose(base.position, base.velocity + base.external_velocity)),
            None => (actor != 0)
                .then(|| self.players.iter().position(|slot| slot.base.id == actor))
                .flatten()
                .map(|k| {
                    let base = &self.players[k].base;
                    pose(base.position, base.velocity + base.external_velocity)
                }),
        };
        let [ax, ay, avx, avy] = actor_pose
            .or_else(|| entity_row.map(|row| [row.x, row.y, row.vx, row.vy]))
            .unwrap_or_default();
        let [x, y, vx, vy] = pose(snapshot.position, snapshot.velocity);
        let related = snapshot.ground.entity_id;
        let mut related_actor = 0;
        if related != 0
            && let Some(mark) = self.events.blinks.iter().find(|mark| mark.entity_id == related)
            && (self.blink_target_hidden(mark.target, mark.entity_id)
                || self.event_tick().wrapping_sub(mark.last_hidden) <= BLINK_WINDOW)
        {
            flags |= DEATH_RELATED_BLINKED;
            related_actor = mark.blinker;
        }
        let event = CEvent {
            kind: EVENT_DEATH,
            tick: self.event_tick(),
            subject,
            actor,
            entity: entity_row.map_or(0, |row| row.id),
            source_ability,
            cause,
            flags,
            related,
            related_actor,
            related_tick: if related != 0 { snapshot.ground.tick } else { 0 },
            reserved: 0,
            x,
            y,
            vx,
            vy,
            ax,
            ay,
            avx,
            avy,
        };
        (event, source)
    }

    /// An eating slime's chew: its own body did it, with no ability.
    pub(super) fn eater_source(&self, eater: BodyRef) -> KillSource {
        let entity = match eater {
            BodyRef::Body(i) => self.players.get(i).map(|slot| slot.body_entity),
            BodyRef::Clone(i) => self.abilityless_clones.get(i).and_then(|clone| clone.entity),
            BodyRef::None => None,
        };
        KillSource { entity, actor: eater, cause: DEATH_EATEN, ..KillSource::default() }
    }

    /// A Black Hole's swallow.
    pub(super) fn hole_source(hole: &BlackHoleState) -> KillSource {
        KillSource { entity: hole.entity, cause: DEATH_CONTACT, ability: ABILITY_BLACK_HOLE, ..KillSource::default() }
    }

    /// Records a BLINK hit by `caster` on `target`, whose collider row is
    /// `entity`.
    pub(super) fn record_blink(&mut self, caster: usize, target: BodyRef, entity: usize, flags: u32) {
        let object = match target {
            BodyRef::Body(i) => BlinkTarget::Body(i),
            BodyRef::Clone(i) => BlinkTarget::Clone(i),
            BodyRef::None => match self.platform_entities.iter().position(|&candidate| candidate == entity) {
                Some(p) if flags & BLINK_PLATFORM != 0 => BlinkTarget::Platform(self.platforms[p].id),
                _ => BlinkTarget::Object(entity),
            },
        };
        let Some(row) = self.entities.get(entity).copied() else {
            return;
        };
        let subject = match target {
            BodyRef::Body(i) => self.players[self.players[i].owner].base.id,
            BodyRef::Clone(i) => self.players[self.abilityless_clones[i].owner].base.id,
            BodyRef::None => row.owner,
        };
        let [x, y, vx, vy] = match self.body_base(target) {
            Some(base) => pose(base.position, base.velocity + base.external_velocity),
            None => [row.x, row.y, row.vx, row.vy],
        };
        let shooter = &self.players[caster].base;
        let [ax, ay, avx, avy] = pose(shooter.position, shooter.velocity + shooter.external_velocity);
        let actor = shooter.id;
        let now = self.event_tick();
        self.events.out.push(CEvent {
            kind: EVENT_BLINK,
            tick: now,
            subject,
            actor,
            entity: row.id,
            source_ability: row.reserved[0],
            cause: 0,
            flags,
            related: 0,
            related_actor: 0,
            related_tick: 0,
            reserved: 0,
            x,
            y,
            vx,
            vy,
            ax,
            ay,
            avx,
            avy,
        });
        let mark = BlinkMark { entity_id: row.id, target: object, blinker: actor, last_hidden: now };
        match self.events.blinks.iter_mut().find(|old| old.entity_id == row.id) {
            Some(old) => *old = mark,
            None => self.events.blinks.push(mark),
        }
    }

    /// After a command's last observation: entity ids to published ones.
    pub(super) fn publish_events(&mut self) {
        if self.events.out.is_empty() {
            return;
        }
        let ids = &self.observed_ids;
        let publish = |id: u32| if id == 0 { 0 } else { ids.get(id as usize).copied().unwrap_or(0) };
        for event in &mut self.events.out {
            event.entity = publish(event.entity);
            event.related = publish(event.related);
        }
    }
}
