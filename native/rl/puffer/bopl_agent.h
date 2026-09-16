#ifndef BOPL_AGENT_H
#define BOPL_AGENT_H
/* Agent shim: the TRAINING decision loop of bopl_puffer.h (observation
 * encoder, nine-tick observation delay, four-tick decision cadence, hold
 * latch, action mask, own-action tail, scripted bots) plus the CPU policy
 * evaluator (policy_eval.c) and a categorical sampler, driven by states a
 * RUNNING GAME hands it once per simulation tick instead of by a world the
 * shim steps itself. The real Unity game (RlAgent.cs) and the fidelity check
 * (research/check_agent_fidelity.py) both go through this one library.
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
 * bopl_bots.h (archer|pursuer|evasive|mixed[-v2]-easy|medium|hard). The
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

#ifdef __cplusplus
}
#endif
#endif
