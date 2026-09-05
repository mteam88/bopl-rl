#ifndef BOPL_H
#define BOPL_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Version 1: Level1, one through six equal-size slimes, jump and dash. All int64_t game
 * values are raw signed Q32.32. No Unity installation is needed at runtime.
 * Handles own their memory. Different handles may run on different threads;
 * calls using the same handle must be externally serialized. */
typedef struct BoplWorld BoplWorld;
typedef struct { uint8_t angle; uint8_t buttons; } BoplAction;
/* angle 0 = neutral; 1..255 = (angle-1)*2*pi/255. buttons: jump=1, dash=2. */
typedef struct {
    int64_t x, y, vx, vy, rotation, local, grounded_speed, cooldown;
    uint32_t id, ground_id;
    uint8_t alive, grounded, dashing, reserved;
} BoplObservation;
enum BoplStatus {
    BOPL_OK=0, BOPL_INVALID_ARGUMENT=1, BOPL_UNSUPPORTED=2,
    BOPL_EPISODE_ENDED=3, BOPL_BUFFER_TOO_SMALL=4, BOPL_INTERNAL_ERROR=5
};
uint32_t bopl_abi_version(void);
BoplWorld *bopl_create(void); /* NULL on failure. Deterministic post-spawn reset. */
/* Additive multiplayer entry point; default bopl_create still creates two.
 * Only player counts 1..6 are supported. Other values return NULL. */
BoplWorld *bopl_create_players(size_t players);
size_t bopl_player_count(const BoplWorld *world); /* 0 for NULL. */
void bopl_destroy(BoplWorld *world); /* NULL is allowed; otherwise exactly once. */
int32_t bopl_reset(BoplWorld *world);
int32_t bopl_step(BoplWorld *world, const BoplAction *actions, size_t count);
int32_t bopl_observe(const BoplWorld *world, BoplObservation *out, size_t count);
/* Returns 0 or 1; -1 for a NULL handle. Episode ends at the first player death. */
int32_t bopl_terminal(const BoplWorld *world);
/* JSON includes map geometry and all Rust state. required includes trailing NUL.
 * Query size with out=NULL, capacity=0. Buffer-too-small leaves out untouched. */
int32_t bopl_state_json(const BoplWorld *world, char *out, size_t capacity, size_t *required);
/* All non-NULL pointers must be valid and suitably aligned for their types.
 * Output memory must not alias the handle or input memory. Failed steps leave
 * state unchanged. No C++ exceptions or Rust panics cross this interface. */
#ifdef __cplusplus
}
#endif
#endif
