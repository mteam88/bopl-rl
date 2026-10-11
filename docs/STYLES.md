# Style exploiters: events, the style spec, and the evaluator

Approved 2026-10-11 (Matthew). A **style exploiter** is a league exploiter that
gets a small, fading training bonus for playing a human style. It joins the league
by beating main (>70%) like any exploiter, or times out. The win/loss reward is
unchanged. Styles come from human replays (`style-discovery/`) and from Matthew.

Two parts, built in parallel against this contract:

1. **Game events** (sim): a per-step stream of things that happened, starting with
   deaths and blinks, from the Rust sim through the ABI.
2. **Style spec + evaluator** (trainer): TOML style files compiled to a small
   program that the C environment runs every decision over the ABI state, the
   event stream and history windows, adding its output to the PPO reward.

Design rule: **nothing is style-specific in C or Rust.** A new style is a TOML
file. A new kind of game fact is one new event kind or state field that every
style can then use.

## 1. Game events (ABI addition)

The state layout does not change. Events are a separate accessor, so every existing
consumer keeps working:

```c
/* Events the last step/step_sampled/reset produced, in the order the game
 * produced them. Valid until the next command; NULL with count 0 when there were
 * none. A backend without events (the socket .NET game) returns NULL and
 * bopl_rl_event_support() == 0. */
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
enum BoplRlDeathFlags { BOPL_DEATH_ACTOR_IN_FORM=1, BOPL_DEATH_LAST_BODY=2, BOPL_DEATH_RELATED_BLINKED=4 };
const BoplRlEvent *bopl_rl_events(const BoplRl *world, uint32_t *count);
int bopl_rl_event_support(const BoplRl *world);
```

- **DEATH**: one per body death (`player_kill`). `actor` is the credited killer
  (same rule as `kills`), `entity` the entity whose contact or effect killed the
  body, `source_ability` its ability (or the actor's form ability when the actor's
  own form body did it, with ACTOR_IN_FORM set). `cause` comes from the kill route.
  Ring-out means scene bounds or a fall. `related` is the victim's last ground or
  contact entity before death, with its tick in `related_tick`. If that entity is
  hidden by a Blink gun tunnel, or was within the last N ticks, then
  RELATED_BLINKED is set and `related_actor` is the blinker.
- **BLINK**: a Blink gun hit (`ShootQuantum.BlinkObject`). `actor` is the
  shooter, `entity` the blinked object, `subject` its owner (0 for map), and
  `flags & 1` is set if it was a player.
- **Adding a kind** = a new enum value plus its emission point. Consumers ignore
  kinds they don't know.

Fidelity:
- Events must not change the simulation: the replay and divergence gates have to
  stay byte-identical.
- Implementing the C# reference too, and A/B-comparing events on the replay
  corpus, is preferred. If that is too costly, mark the .NET backend unsupported
  and spot-check the Rust events on the human replays instead.

## 2. Style spec (TOML)

```toml
# styles/drill_snipe.toml
name = "drill_snipe"
kits = [[26, 3, 2], [2, 26, 3]]       # loadouts the exploiter trains on (optional)
fade = { start = 0.0, end = 0.7 }      # bonus scale 1 -> 0 over this fraction of the turn

[[term]]
on = "death"                           # event term: evaluated per event
when = "ev.actor == self and ev.source_ability == ability.Drill and hypot(ev.avx, ev.avy) >= 20 and since_form_entry(self) >= 0.5 * 60"
value = "0.08"

[[term]]                               # per-decision term
when = "ready_count(self) >= 1"
value = "8e-5"
```

- **Expressions** use a small language: numbers, `and/or/not`, comparisons, `+ - * /`,
  `min/max/abs/sqrt`, and named ability ids (`ability.Drill`) and categories.
- **Names available:**
  - `self`, `opp` (the nearest living opponent, or `for opp in opponents` with
    `sum`/`any`).
  - Every ABI field: `self.x`, `self.flags`, `self.deaths`, `body(self).vx`,
    `body(self).ability_flags[slot]`, `header.episode_ticks`...
  - Inside an event term, `ev.*` fields.
- **Aggregates over entities:** `count(entities where owner == self and flags & EXPLOSIVE and dist(opp) < 6)`,
  `min_dist`, `coverage(opp, radius, sectors, filter)`. Coverage is the fraction
  of angular sectors around `opp` that contain a matching hazard; `sources(...)`
  counts the distinct source abilities among them.
- **Categories:** `ready(p, "mobility")` and `ready_count(p, "offense")` read
  `styles/categories.toml`, which maps ability id to a category list. Matthew edits it.
- **History windows:** `since(event-or-condition)` in ticks, `count_over(N, cond)`,
  `max_over(N, expr)`, `ema(tau, expr)`, `prev(expr)`. Each one is a named
  ring-buffer slot that the compiler allocates.
- **Units:** world units, ticks (60/s), speeds in units/s.

## 3. Evaluator

- `train/style_compile.py` parses the TOML into a flat program: a constants
  table, history slots, and per-term bytecode for a stack VM. It refuses unknown
  names and type errors with a precise message. It writes `<run>/style.bin` plus a
  readable dump, and records it and its hash in `config.json`.
- The C side, `train/bopl_style.h`, is included by `bopl_puffer.h`.
  - `bopl_style_step(program, state, events, n, player, slots) -> float` runs once
    per learner decision per player, right where the kill shaping is added in
    `bopl_step_finish_with`.
  - It allocates nothing and uses no floats beyond `float`.
  - Its cost is bounded: under 2% of env step time with three styles active, measured with `make env-bench`.
- `train.py --style styles/x.toml` compiles the spec and loads it into every env.
  The fade scale is set from Python each update with a binding setter. Its
  per-term payouts go into telemetry, so the dashboard shows what is paid out.
- An absent style means zero cost and an identical reward. This is checked by the
  routing and reward checks staying byte-identical.
- **Measurement:** the same compiled program runs offline over the replay dumps
  (`eval/style_score.py`, a Python interpreter of the same bytecode). That is used
  to measure human, bot and exploiter style rates and to validate the evaluator:
  the C and Python runs must agree on the dumps.

## 4. League loop

- New agent `style-exploiter` in `league_loop.py`: its own slot list, cycling
  through `[style-exploiter] styles = [...]`. It trains with `--style` and
  `--historical-every 1` against the league (PFSP). It is added like a league
  exploiter, and its name carries the style.
- Ratings add `style_rates` per member by running `eval/style_score.py` over its
  rating games.
