#ifndef BOPL_AGENT_H
#define BOPL_AGENT_H
/* Agent shim: the TRAINING decision loop of bopl_puffer.h (observation
 * encoder, nine-tick observation delay, four-tick decision cadence, hold
 * latch, action mask, own-action tail, scripted bots) plus the CPU policy
 * evaluator (policy_eval.c) and a categorical sampler, driven by states a
 * RUNNING GAME hands it once per simulation tick instead of by a world the
 * shim steps itself. The real Unity game (RlAgent.cs) and the fidelity check
 * (eval/check_agent_fidelity.py) both go through this one library.
 *
 * Usage, per match round:
 *   a = bopl_agent_open(spec, players, player, error, capacity);
 *   bopl_agent_reset(a, round_seed);
 *   for every simulation tick after the warmup (header.episode_ticks = ticks
 *   already run since the warmup ended, so the first call carries 0):
 *       bopl_agent_tick(a, state, &action);   // action: the input for THIS tick
 *   bopl_agent_close(a);
 *
 * The state passed at tick e is the game state S_e after e episode ticks
 * (RlObservation.cs writes exactly the BoplRlState wire format; the _bytes
 * entry point decodes it). The returned action is the one training would
 * have applied to tick e+1: decision k = e/4 is made at e = 4k from the
 * published (delayed) observation and held for e = 4k..4k+3, as
 * research_step_begin/tick/finish do in eval_binding.c; the observation for
 * the next decision is sampled from S_{4k+3} and published, nine ticks late,
 * at e = 4k+4. Everything else (rewards, telemetry counters, masks) runs
 * through the same code, so a caller can read the observation, mask and
 * logits it decided on and compare them with the research environment.
 *
 * spec is either a checkpoint path "<...>/policy-%09d.bin" (its "<path>.json"
 * sidecar is loaded and honoured: architecture, hidden, layers, action sizes,
 * observation delay, episode length, action repeat, and the loadout the
 * checkpoint trained with) or a scripted bot name "<kind>-<level>" from
 * bopl_bots.h (archer|pursuer|evasive|mixed[-v2]|rusher-easy|medium|hard). The
 * search bot (kind 5) forks the sim per decision and is refused.
 *
 * A third form, "<...>.roster", is a text file of such specs (one per line,
 * optional weight after whitespace, '#' comments; relative checkpoint paths
 * resolve against the roster's directory). All members are opened at once
 * and every bopl_agent_reset draws one for the round, weighted, from the
 * round seed; the draw is appended as one JSON line {"seed","member","spec",
 * "id"} to $BOPL_AGENT_ROSTER_LOG (default "<roster>.log") so a session's
 * rounds can be attributed afterwards (Tools/play_vs_agent_results.py).
 * Members must share loadout, warmup and episode length: the game reads
 * those once per match. bopl_agent_id stays "roster:<name>";
 * bopl_agent_member_id is the current member's id. Rating sessions for a
 * human (Tools/rating_roster.py) use this to spread ~100 rounds over a pool.
 *
 * Sampling: each head (aim 256, buttons 16, hold 5) is drawn from the softmax
 * of its masked logits (disallowed -> -inf), the distribution evaluate.py's
 * torch.multinomial draws from; the draw uses this library's seeded
 * generator, so a match is reproducible from (spec, round seed, states).
 * Checkpoints with a Fourier aim block or an aim block per button combo
 * (sidecar aim_fourier / aim_ar, bopl_aim.h) are materialised to the same
 * 256 aim logits; with aim_ar the buttons are drawn before the aim.
 * Not thread-safe per handle; handles are independent. */
#include "bopl_rl.h"
#include "bopl_shadow.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct BoplAgent BoplAgent;

uint32_t bopl_agent_abi_version(void);  /* 1 */
/* players: players in the match (1..6); player: index of the agent's player
 * (0-based, the order of BoplRlState.players). Returns NULL and writes error. */
BoplAgent *bopl_agent_open(const char *spec, int players, int player, char *error, size_t capacity);
/* New round: recurrent state, latch, delay history and the sampler/bot
 * generators start over from seed. Required before the first tick. */
void bopl_agent_reset(BoplAgent *agent, uint32_t seed);
/* One simulation tick. Returns 0 with the action to apply to this tick, 1
 * when the episode has ended (action neutral), -1 on a misuse (no reset,
 * episode_ticks not advancing by one, players mismatch). */
int bopl_agent_tick(BoplAgent *agent, const BoplRlState *state, BoplRlAction *out);
/* The same on the RlObservation.cs wire format (header, players, entities,
 * slimes, telemetry back to back). -2 on a malformed record. */
int bopl_agent_tick_bytes(BoplAgent *agent, const void *bytes, size_t length, BoplRlAction *out);
void bopl_agent_close(BoplAgent *agent);

/* Configuration the agent runs with (from the sidecar, or the training
 * defaults for bots): loadout slot 0..2, warmup ticks, action repeat,
 * observation delay, episode length. */
int bopl_agent_ability(const BoplAgent *agent, int slot);
int bopl_agent_warmup_ticks(const BoplAgent *agent);
int bopl_agent_repeat(const BoplAgent *agent);
int bopl_agent_observation_delay(const BoplAgent *agent);
int bopl_agent_max_episode_ticks(const BoplAgent *agent);
/* "policy:<weights sha256>", "bot:<kind>-<level>" or "roster:<name>"; "" for NULL. */
const char *bopl_agent_id(const BoplAgent *agent);
/* The id deciding this round: the roster's drawn member, else the agent itself. */
const char *bopl_agent_member_id(const BoplAgent *agent);
int bopl_agent_is_policy(const BoplAgent *agent);

/* Introspection of the latest decision (valid after a tick at a decision
 * boundary): the published observation row and mask the policy saw, the raw
 * decoder output (BOPL_LOGITS logits then the value), the sampled head
 * indices (aim, buttons, hold), and the count of decisions so far. The
 * logits are the game logits the decision was drawn from (with aim_ar: the
 * aim block of the sampled button combo). */
const float *bopl_agent_observation(const BoplAgent *agent);
const unsigned char *bopl_agent_mask(const BoplAgent *agent);
const float *bopl_agent_logits(const BoplAgent *agent);
void bopl_agent_decision(const BoplAgent *agent, int out[3]);
int bopl_agent_decisions(const BoplAgent *agent);
int bopl_agent_obs_size(void);
/* Width of this agent's observation row (bopl_agent_observation): the policy's
 * layout, BOPL4_OBS_SIZE for v4 checkpoints, else bopl_agent_obs_size(). */
int bopl_agent_observation_size(const BoplAgent *agent);
int bopl_agent_logits_size(void);

/* Forced decision (human behaviour cloning, eval/human_dataset.py): the next
 * policy decision plays (angle, buttons, hold) instead of a sample; the
 * policy still runs, so bopl_agent_logits is its distribution on that row.
 * The decision is made legal under the mask first (a latch keeps its held
 * buttons and hold 0; a bow held since the last decision is released);
 * bopl_agent_forced_fixed reports what was changed (bit 8: latched buttons
 * differed, 16: latched hold, 32: buttons masked, 64: hold masked). */
void bopl_agent_force(BoplAgent *agent, int angle, int buttons, int hold);
int bopl_agent_forced_fixed(const BoplAgent *agent);

/* The round's map (BoplRlConfig.map: the replay setup's level byte). The
 * v4 row tells the policy whether it is on a space map, which only the map
 * index says; without this call the shim assumes map 0 (not space). Persists
 * until changed; call before the round's first tick. */
void bopl_agent_set_map(BoplAgent *agent, uint32_t map);

/* Decision-time search in the game (docs/INGAME_SEARCH.md): every decision
 * of the agent is chosen by train/bopl_search.h on forks of `shadow`, a
 * bopl_shadow.h world the caller keeps in lockstep with the game (begin,
 * check S_e before the agent's tick e, step after it). budget is
 * eval/search_policy.py's "k,n,depth[,h][,mix=M][,vs=V][,latched=0]"
 * (default in-game budget 8,1,1). Policy v4 checkpoints only (a roster:
 * every v4 member; others stay raw). Every player's row becomes the
 * checkpoint's layout: the other players (humans) are modelled by the same
 * checkpoint, their decisions mapped from the packets the shadow was stepped
 * with (last tick of the decision: aim, the four action buttons, shortest
 * hold). Rollouts end at the agent's episode limit with the timeout outcome.
 * A round falls back to the raw policy for good, logging why on stderr and in
 * bopl_agent_search_note, when the shadow is not tracking (desync, refused,
 * not begun), is not at the decision's tick, the episode cap is reached, or
 * the search fails (a rollout's sim error included: it does not abort the
 * game). Call after open, before the first round. 0, or -1 with error. */
int bopl_agent_search(BoplAgent *agent, BoplShadow *shadow, const char *budget, char *error, size_t capacity);
/* 1 while this round's decisions are searched (0: off, or fell back). */
int bopl_agent_search_active(const BoplAgent *agent);
/* Why this round fell back to the raw policy ("" while searching). */
const char *bopl_agent_search_note(const BoplAgent *agent);
/* This round's search counters into out[BOPL_AGENT_SEARCH_STATS]: policy
 * decisions, chosen by the search, chosen raw (fallback), search seconds,
 * max seconds per decision, p50, p95, p99 seconds, rollouts, rollout decision
 * steps, forward rows, searched (more than one candidate), fallback episode
 * tick (-1 none), active. Returns the count written (0: no search), or minus
 * the capacity needed. */
enum { BOPL_AGENT_SEARCH_STATS=14 };
int bopl_agent_search_stats(const BoplAgent *agent, double *out, int capacity);

#ifdef __cplusplus
}
#endif
#endif
