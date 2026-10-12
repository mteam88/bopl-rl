#ifndef BOPL_SHADOW_H
#define BOPL_SHADOW_H
/* In-game shadow sim (docs/SHADOW.md): milestone 1 of decision-time search
 * inside the agent shim. The bopl_rl.h ABI has no "set state", so a search
 * that forks the Rust sim (train/bopl_search.h) needs a sim world that is
 * already in the game's state. The shadow is that world: it is reset from the
 * round's own BOPLREP2 setup (ReplaySession.Recording after the first tick)
 * and stepped with every player's applied input packet, tick for tick, while
 * the game runs. Each tick the game's observation (the RlObservation.cs wire
 * format RlAgent already hands bopl_agent_tick_bytes) is compared with the
 * shadow's state using the replay differential's comparator
 * (sim/abi/bopl_rl_compare.h). The first mismatch of a round is reported and
 * the shadow stops for that round. bopl_agent_search (bopl_agent.h) runs the
 * decision-time search on forks of this world (docs/INGAME_SEARCH.md).
 *
 * Per round (RlAgent.cs drives this):
 *   bopl_shadow_begin(s, archive, length, warmup)  after level tick 1: setup
 *                                  + source-hash check, async reset (the
 *                                  shadow runs its own zero-input warmup)
 *   bopl_shadow_check_bytes(s, obs, len)  game state S_e vs shadow, e >= 0
 *   bopl_shadow_step(s, actions, players) the packets the game applied to the
 *                                  tick that produced S_{e+1}, PlayerList order
 *   bopl_shadow_end(s)             the game's round is over (after the
 *                                  terminal state was checked)
 *
 * The library is the q32 Rust sim (bit-exact Q32.32; `make rust-q32`), loaded
 * in-process: `library` argument, else $BOPL_AGENT_SHADOW_LIB, else
 * libbopl_sim_q32.so / .dylib next to this shim.
 *
 * Source check: the round's archive carries the game's pinned source hash
 * (ReplaySession.SourceHash). It must equal bopl_rl_source_hash of the shadow
 * library. The Rust port reports a placeholder hash (native/src/rollout/
 * mod.rs ROLLOUT_SOURCE_HASH); for it the game's hash must equal the source
 * the port was checked against, BOPL_SHADOW_RUST_SOURCE below (override with
 * $BOPL_AGENT_SHADOW_SOURCE=<hash>, or "any" to skip). A mismatch refuses the
 * round with a clear report and status BOPL_SHADOW_FAILED.
 *
 * The game's Player.Kills/Deaths count the whole match while the shadow's
 * start at zero each round, so kills and deaths are compared relative to the
 * round's first checked state. Likewise the game's prefab HierarchyNumbers
 * (>= 100000000) carry the prefabs of earlier rounds (Updater.PreLevelLoad
 * keeps prefabsAdded); the constant offset is learned at the first check.
 *
 * Comparison options (environment, read at open):
 *   BOPL_AGENT_SHADOW_TOL        numeric tolerance, raw Q32.32 units (0)
 *   BOPL_AGENT_SHADOW_VISIBLE    1 = only what a policy can read
 *   BOPL_AGENT_SHADOW_STRICT     1 = no inactive-prefab ENABLED-bit exclusion
 *   BOPL_AGENT_SHADOW_IGNORE     generalized fields to skip ("entity[].flags,...")
 *   BOPL_AGENT_SHADOW_FLAGS      header.flags bits compared (default 1, TERMINAL:
 *                                RlAgent sets STARTED after the warmup, the
 *                                sims during it)
 *   BOPL_AGENT_SHADOW_TELEMETRY  1 = compare telemetry[] (RlAgent never counts
 *                                it, so off by default)
 *   BOPL_AGENT_SHADOW_LOG        the library's own log (/tmp/bopl-shadow.log)
 * Not thread-safe per handle. */
#include "bopl_rl.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define BOPL_SHADOW_RUST_SOURCE "d0562ac89d03d46f24f122a08f4f5366c675cc94a015a4ccaf7adc124af089b7"

typedef struct BoplShadow BoplShadow;
enum BoplShadowStatus {
    BOPL_SHADOW_OFF = 0,       /* no round begun */
    BOPL_SHADOW_TRACKING = 1,  /* every compared tick matched */
    BOPL_SHADOW_DESYNC = 2,    /* a compared tick differed; stopped for the round */
    BOPL_SHADOW_ENDED = 3,     /* the round ended in sync */
    BOPL_SHADOW_FAILED = 4     /* refused or broken (source mismatch, sim error) */
};

/* Opens the shadow library's world (no round yet). NULL with error text. */
BoplShadow *bopl_shadow_open(const char *library, char *error, size_t capacity);
/* New round from a BOPLREP2 archive (only its setup is read; frames are
 * ignored). warmup: the zero-input warmup ticks the game runs (RlAgent's
 * WarmupTicks). Returns the status: TRACKING, or FAILED with the reason in
 * bopl_shadow_report. The reset runs on a thread until the first check/step. */
int bopl_shadow_begin(BoplShadow *shadow, const void *archive, size_t length, uint32_t warmup);
/* Compares the game's state (wire format) with the shadow's current state.
 * Returns the status after the comparison. -2 on a malformed record. */
int bopl_shadow_check_bytes(BoplShadow *shadow, const void *bytes, size_t length);
int bopl_shadow_check(BoplShadow *shadow, const BoplRlState *game);
/* Advances the shadow one tick with all players' applied packets. Returns the
 * status (a sim error is FAILED). Stepping past the shadow's own terminal
 * state is a no-op; the next check shows the game did not end there. */
int bopl_shadow_step(BoplShadow *shadow, const BoplRlAction *actions, size_t count);
/* The game's round is over: TRACKING becomes ENDED. Returns the status. */
int bopl_shadow_end(BoplShadow *shadow);
int bopl_shadow_status(const BoplShadow *shadow);
const char *bopl_shadow_status_name(int status);
/* One-line summary plus, after a desync, every differing field of that tick.
 * "" before anything happened. Valid until the next call on the handle. */
const char *bopl_shadow_report(const BoplShadow *shadow);
/* Ticks compared this round, and the shadow's episode tick. */
uint32_t bopl_shadow_compared(const BoplShadow *shadow);
uint32_t bopl_shadow_episode_tick(const BoplShadow *shadow);
/* The packets of the last bopl_shadow_step this round (episode tick
 * bopl_shadow_episode_tick - 1), PlayerList order; returns their count, 0
 * before the first step or when capacity is short. */
size_t bopl_shadow_packets(const BoplShadow *shadow, BoplRlAction *out, size_t capacity);
/* The shadow world (for the search milestone: bopl_rl_fork it) and its state;
 * NULL unless the round is TRACKING with the reset finished. */
BoplRl *bopl_shadow_world(BoplShadow *shadow);
const BoplRlState *bopl_shadow_state(BoplShadow *shadow);
const char *bopl_shadow_library(const BoplShadow *shadow);
void bopl_shadow_close(BoplShadow *shadow);

#ifdef __cplusplus
}
#endif
#endif
