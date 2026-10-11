#ifndef BOPL_RL_H
#define BOPL_RL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Version 3, Linux little-endian hosts. Owns an isolated, persistent headless
 * game process. All fixed-point values are signed Q32.32. Serialize calls for
 * one handle; separate handles can run concurrently. */
typedef struct BoplRl BoplRl;
enum BoplRlStatus { BOPL_RL_OK=0, BOPL_RL_ARGUMENT=1, BOPL_RL_IO=2, BOPL_RL_GAME_ERROR=3, BOPL_RL_ENDED=4 };
enum BoplRlFlags { BOPL_RL_TERMINAL=1, BOPL_RL_TRUNCATED=2, BOPL_RL_STARTED=4, BOPL_RL_FAILED=8 };
enum BoplRlEntityFlags {
    BOPL_ENTITY_ACTIVE=1, BOPL_ENTITY_ENABLED=2, BOPL_ENTITY_DESTROYED=4,
    BOPL_ENTITY_TRIGGER=8, BOPL_ENTITY_COMPOSITE=16, BOPL_ENTITY_DETACHED=32,
    BOPL_ENTITY_TIMER_KNOWN=64, BOPL_ENTITY_PICKUP=128, BOPL_ENTITY_PROJECTILE=256,
    BOPL_ENTITY_SPARK_NODE=512, BOPL_ENTITY_FLAMMABLE_SMOKE=1024,
    BOPL_ENTITY_EXPLOSIVE=2048, BOPL_ENTITY_HITBOX=4096,
    BOPL_ENTITY_CONTACT_DAMAGE=8192, BOPL_ENTITY_CONTACT_DAMAGE_KNOWN=16384
};
enum BoplRlSlimeFlags {
    BOPL_SLIME_ACTIVE=1, BOPL_SLIME_SPAWNED=2, BOPL_SLIME_IN_ABILITY=4,
    BOPL_SLIME_SUPPRESS_INPUT=8, BOPL_SLIME_SIMULATED=16, BOPL_SLIME_ABILITYLESS=32,
    BOPL_SLIME_PICKUP_SLOT_SHIFT=6, BOPL_SLIME_PICKUP_SLOT_MASK=3
};
typedef struct { uint8_t angle, reserved; uint16_t buttons; } BoplRlAction;
/* angle: 0 neutral, 1..255 directional. buttons: jump=1, abilities=2/4/8,
 * start=16, select=32, W/A/S/D=64/128/256/512. reserved must be zero. */
/* Ability 0 is the catalog's menu placeholder and is rejected as a loadout. */
typedef struct { uint32_t color, team, keyboard, abilities[3]; } BoplRlPlayerConfig;
typedef struct {
    uint32_t seed, map, players, slots, stocks, mode, warmup_ticks, max_episode_ticks;
    BoplRlPlayerConfig player[6];
} BoplRlConfig;
typedef struct {
    uint32_t id, team, flags, alive_bodies, stocks, kills, deaths, cause_of_death;
    int64_t x, y, scale, invisible_time, ability_time;
} BoplRlPlayer;
typedef struct {
    /* reserved[0]: source/granted ability id. reserved[1]: nominal ticks to
     * the next event when BOPL_ENTITY_TIMER_KNOWN is set. */
    uint32_t id, kind, layer, owner, flags, hierarchy, reserved[2];
    int64_t x, y, vx, vy, right_x, right_y, up_x, up_y, radius, rotation, angular_velocity, inverse_mass, clock_rate;
} BoplRlEntity;
typedef struct {
    uint32_t id, owner, flags, state;
    int64_t x, y, vx, vy, rotation, scale;
    uint32_t abilities[3], ability_flags[3];
    int64_t cooldowns[3], timers[3];
} BoplRlSlime;
typedef struct {
    uint32_t version, tick, episode_ticks, flags, player_count, entity_count, warmup_ticks, slime_count;
    int64_t simulation_time, player_time_scale;
    int64_t simulation_time_since_level_loaded;
    uint32_t sudden_death, reserved;
} BoplRlHeader;
/* Episode-cumulative event counters, excluding warmup. Research telemetry only;
 * not policy observations. Deaths follow the game's credited killer. */
typedef struct {
    uint32_t ability_activations[3], arrows_fired, arrow_hits;
    uint32_t deaths_environment, deaths_opponent, deaths_self, deaths_other;
} BoplRlTelemetry;
typedef struct {
    BoplRlHeader header;
    BoplRlPlayer players[6];
    const BoplRlEntity *entities;
    const BoplRlSlime *slimes;
    BoplRlTelemetry telemetry[6];
} BoplRlState;

uint32_t bopl_rl_abi_version(void);
void bopl_rl_default_config(BoplRlConfig *config, uint32_t players);
/* Paths must be absolute. Bootstrap replay selects the initial game bootstrap;
 * reset selects the actual RL configuration. timeout_ms covers each operation.
 * Error text is written to error (when non-NULL and capacity > 0).
 * An executable ending in ".so" is a NativeAOT library exporting this ABI
 * (sim/Native): it is loaded into this process, one private copy per handle,
 * and stepped by direct calls instead of a worker process and socket. */
int bopl_rl_in_process(const char *executable);
BoplRl *bopl_rl_open(const char *executable, const char *bootstrap_replay,
                     const char *log_path, uint32_t timeout_ms, char *error, size_t capacity);
int bopl_rl_reset(BoplRl *world, const BoplRlConfig *config);
/* bopl_rl_reset split in two: begin sends the configuration and returns while
 * the game loads the scene and runs its warmup ticks; finish waits for the
 * reply and installs the state. Between the two, the state is invalid and no
 * other command may be issued; close finishes a pending reset itself. */
int bopl_rl_reset_begin(BoplRl *world, const BoplRlConfig *config);
int bopl_rl_reset_finish(BoplRl *world);
int bopl_rl_reset_pending(const BoplRl *world);
int bopl_rl_step(BoplRl *world, const BoplRlAction *actions, size_t count, uint32_t repeat);
/* One roundtrip for two consecutive phases of the same actions: `first` ticks,
 * then, unless the episode ended, `second` more ticks (first+second <= 16).
 * bopl_rl_sample returns the state after the first phase and bopl_rl_state the
 * final state; both stay valid until the next command. When the episode ends
 * during the first phase, the sample and the final state are identical, like
 * two separate step calls. A game build without this command is a
 * BOPL_RL_GAME_ERROR ("Unknown RL command"); there is no fallback. */
int bopl_rl_step_sampled(BoplRl *world, const BoplRlAction *actions, size_t count, uint32_t first, uint32_t second);
const BoplRlState *bopl_rl_sample(const BoplRl *world);
/* Cached state; valid until the next step/reset/close. NULL before reset or after
 * a transport/game error. This does not cross into the game process. */
const BoplRlState *bopl_rl_state(const BoplRl *world);
/* Lossless BOPLREP2 setup + warmup + actual tick inputs, including all six
 * players. Bytes are borrowed until the next command or close. */
int bopl_rl_replay(BoplRl *world, const void **bytes, size_t *length);
/* Lookahead copy: a handle whose game is a private snapshot of world's (same
 * state and random stream, so the same actions give the same future); step,
 * state and close work on it like on any handle, replay/reset do not apply.
 * `reuse` (a previous fork, or NULL) is refilled instead of allocating. NULL
 * when the backend cannot fork: only in-process libraries exporting
 * bopl_rl_fork (the Rust sim) can; the socket/.NET game cannot. */
BoplRl *bopl_rl_fork(const BoplRl *world, BoplRl *reuse);
const char *bopl_rl_error(const BoplRl *world);
const char *bopl_rl_source_hash(const BoplRl *world);
const char *bopl_rl_catalog_hash(const BoplRl *world);
uint32_t bopl_rl_map_count(const BoplRl *world);
uint32_t bopl_rl_ability_count(const BoplRl *world);

/* Game events (docs/STYLES.md): what the last step/step_sampled/reset
 * produced, in the order the game produced them (step_sampled: both phases).
 * Valid until the next command; NULL with count 0 when there were none. A
 * backend without events (the socket .NET game) returns NULL and
 * bopl_rl_event_support() == 0. The state layout does not change.
 *
 * tick is the header.tick of the first state that shows the event. Entity ids
 * are published ids (0 when no observation ever listed the entity). */
#define BOPL_RL_EVENTS 1  /* train/bopl_style.h: this header declares the events ABI */
typedef struct {
    uint32_t kind;            /* BOPL_EVENT_* */
    uint32_t tick;            /* header.tick at the event */
    uint32_t subject;         /* player id the event happened to (victim, blinked owner...) */
    uint32_t actor;           /* player id that caused it; 0 environment/none */
    uint32_t entity;          /* entity id involved (the killing entity, the blinked object), 0 none */
    uint32_t source_ability;  /* ability id behind `entity` or the actor's form, 0 unknown */
    uint32_t cause;           /* kind-specific: deaths use BOPL_DEATH_* */
    uint32_t flags;           /* kind-specific bits */
    uint32_t related;         /* kind-specific entity id (deaths: victim's last contact entity) */
    uint32_t related_actor;   /* kind-specific player id (deaths: who blinked `related`, 0 none) */
    uint32_t related_tick;    /* kind-specific tick (deaths: tick of last contact) */
    uint32_t reserved;
    int64_t x, y, vx, vy;     /* Q32.32: subject's body at the event */
    int64_t ax, ay, avx, avy; /* Q32.32: actor's body (or the entity) at the event */
} BoplRlEvent;
enum BoplRlEventKind { BOPL_EVENT_DEATH=1, BOPL_EVENT_BLINK=2 };
enum BoplRlDeathCause {   /* finer than CauseOfDeath; derived from the kill route */
    BOPL_DEATH_UNKNOWN=0, BOPL_DEATH_CONTACT=1, BOPL_DEATH_PROJECTILE=2, BOPL_DEATH_EXPLOSION=3,
    BOPL_DEATH_BEAM=4, BOPL_DEATH_SQUISHED=5, BOPL_DEATH_EATEN=6, BOPL_DEATH_RING_OUT=7,
    BOPL_DEATH_SELF=8, BOPL_DEATH_OTHER=9
};
/* LAST_BODY: the death eliminated the player (Player.Kill saw its last body
 * and set `out`; a stock or revival respawn does not). RELATED_BLINKED:
 * `related` is in a Blink tunnel, or was within the last 60 ticks. */
enum BoplRlDeathFlags { BOPL_DEATH_ACTOR_IN_FORM=1, BOPL_DEATH_LAST_BODY=2, BOPL_DEATH_RELATED_BLINKED=4 };
/* BLINK flags: bit 0 a player body (the contract's flag); the others refine it. */
enum BoplRlBlinkFlags { BOPL_BLINK_PLAYER=1, BOPL_BLINK_PLATFORM=2, BOPL_BLINK_CLONE=4, BOPL_BLINK_SELF=8 };
const BoplRlEvent *bopl_rl_events(const BoplRl *world, uint32_t *count);
int bopl_rl_event_support(const BoplRl *world);

/* NULL is allowed. Closes and reaps only this handle's worker process group. */
void bopl_rl_close(BoplRl *world);

#ifdef __cplusplus
}
#endif
#endif
