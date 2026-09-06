#ifndef BOPL_RL_H
#define BOPL_RL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Version 2, Linux little-endian hosts. Owns an isolated, persistent headless
 * game process. All fixed-point values are signed Q32.32. Serialize calls for
 * one handle; separate handles can run concurrently. */
typedef struct BoplRl BoplRl;
enum BoplRlStatus { BOPL_RL_OK=0, BOPL_RL_ARGUMENT=1, BOPL_RL_IO=2, BOPL_RL_GAME_ERROR=3, BOPL_RL_ENDED=4 };
enum BoplRlFlags { BOPL_RL_TERMINAL=1, BOPL_RL_TRUNCATED=2, BOPL_RL_STARTED=4, BOPL_RL_FAILED=8 };
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
    uint32_t id, kind, layer, owner, flags, hierarchy, reserved[2];
    int64_t x, y, vx, vy, right_x, right_y, up_x, up_y, radius, rotation, angular_velocity, inverse_mass;
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
 * Error text is written to error (when non-NULL and capacity > 0). */
BoplRl *bopl_rl_open(const char *executable, const char *bootstrap_replay,
                     const char *log_path, uint32_t timeout_ms, char *error, size_t capacity);
int bopl_rl_reset(BoplRl *world, const BoplRlConfig *config);
int bopl_rl_step(BoplRl *world, const BoplRlAction *actions, size_t count, uint32_t repeat);
/* Cached state; valid until the next step/reset/close. NULL before reset or after
 * a transport/game error. This does not cross into the game process. */
const BoplRlState *bopl_rl_state(const BoplRl *world);
/* Lossless BOPLREP2 setup + warmup + actual tick inputs, including all six
 * players. Bytes are borrowed until the next command or close. */
int bopl_rl_replay(BoplRl *world, const void **bytes, size_t *length);
const char *bopl_rl_error(const BoplRl *world);
const char *bopl_rl_source_hash(const BoplRl *world);
const char *bopl_rl_catalog_hash(const BoplRl *world);
uint32_t bopl_rl_map_count(const BoplRl *world);
uint32_t bopl_rl_ability_count(const BoplRl *world);
/* NULL is allowed. Closes and reaps only this handle's worker process group. */
void bopl_rl_close(BoplRl *world);

#ifdef __cplusplus
}
#endif
#endif
