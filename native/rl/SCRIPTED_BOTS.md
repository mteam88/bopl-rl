# SCRIPTED_BOTS.md — hand-coded opponents for training and evaluation

`native/rl/puffer/bopl_bots.h` implements four algorithmic (non-learning)
opponents for the Dash/Bow/Platform loadout: **archer**, **pursuer**,
**evasive** and **mixed**, each at three difficulty levels (easy/medium/hard).
They exist as fixed anchors: strong enough to beat a fresh policy nearly
always and to seriously challenge a well-trained one, and as an optional
training opponent so a learner cannot forget how to fight a competent,
aggressive player.

## Contract

A bot replaces one player's policy inside the C environment. It reads the
**true current `BoplRlState`** — perfect information, no 9-tick observation
delay; deliberate cheating, which is fine for an anchor — but acts through
exactly the policy interface:

- one `(aim byte, button bits, hold index)` decision per four ticks;
- decisions run through the same `bopl_latch`, so a button hold pins its
  buttons for the latch length and an arrow press is a charge that fires when
  the latch expires (the `bopl_write_mask` arrow-release rule);
- a forced-release decision never emits an arrow press (mask parity).

Nothing else changed: not the action space, observation layout, masks, or
rewards. The policy player's rows are untouched; a bot-controlled player's
GPU/action floats are simply ignored (`c_step` asks `bopl_bot_decide`
instead).

## Versioning

Reference opponents are never mutated, only added (the OpenAI Five rule):
every trial, ladder and sweep score ever recorded against `mixed-hard` must
stay comparable with the ones recorded tomorrow. The kind table in
`bopl_bots.h` (`BOPL_BOT_*` ids, `BOPL_BOT_NAMES`) and its Python mirror
`evaluation_protocol.SCRIPTED_KINDS` are therefore append-only: kinds 1–4
(`archer`, `pursuer`, `evasive`, `mixed`) are the frozen **v1** set, and the
code they run — everything in `bopl_bots.h` — does not change again. An
improved or new bot is a *new kind*: append an id (< 16, one nibble per
training bank) and a name of the form `<family>[-vN]` (`mixed-v2`, `search`),
mirror it in `SCRIPTED_KINDS`, and dispatch it from `bopl_bots_v2.h`, which
may call the v1 helpers but never edits them. The public name stays
`kind-level` with the level last (`mixed-v2-hard`, `search-medium`), so
`evaluate.py --scripted-opponents`, `elo_ladder.py --scripted-anchors`,
`train.py --scripted-bots`, the sweep configs and the dashboard label
("Scripted mixed-v2 (hard)") need no per-bot code; the level suffix is
reserved for scripted names (a frozen-opponent manifest may not use it).
The v1 freeze is checked by fixed-seed fingerprints: `check_scripted_bots.py`
plus 20 idle-opponent games per `kind-level` must reproduce the same replay
bytes before and after any change to the bot headers.

## search (kind 5, v2): rollout search on a forked sim

`search-hard` / `search-medium` (`bopl_bots_v2.h`) are model-predictive
control: every decision the bot forks the live game (`bopl_rl_fork`, a
private copy with the same random stream), plays a set of candidate action
sequences on the fork for a horizon and emits the first decision of the best.
Because the rollouts run the real sim, everything v1 approximates is exact
for free — the 13 u dash raycast through projectiles, own-arrow immunity
ending after 5 u, a fire-delayed arrow that does not exist yet as an entity,
the growing platform of a squish, the platform morph's ejection. The bot
needs the Rust sim backend (`BOPL_RL_GAME=.../libbopl_sim.so`); the .NET
game cannot fork and the bot aborts with a message.

**Fork FFI** (`src/rollout/abi.rs::bopl_rl_fork`, `RolloutWorld::fork_from`,
C `bopl_rl_fork(world, reuse)`): refilling a reused fork costs ~1 µs (fresh
allocation ~21 µs), a tick on the fork 0.5–0.6 µs at the 4-tick decision
cadence; a 120-tick rollout including the fork is ~55–70 µs. Determinism was
checked by driving a fork and the real world with identical random inputs to
the end of the game on maps 0–2: zero state mismatches. `data::get()` is a
read-only `OnceLock`, so forks are thread-safe per environment.

**Candidates** (hard ≈ 35, medium ≈ 15): stand, walk ±, jump ±, dash in 8
directions plus toward the foe, dash-then-shoot (the v1 solver fires once the
dash tail ends), shots for each charge (v1 solver aims terrain-checked and
flat, ±3-step perturbations, random aims; horizon = charge + fire delay +
flight), platform (horizon covers the ~72-tick morph and ejection), and six
random sequences (hard). While latched only the aim varies (an arrow charge
searches around `bot_charge_aim`).

**Scoring**: own death −1 (+0.25·t/H), foe death +1 (−0.25·t/H), both −1;
otherwise a heuristic on the end state: a certified shot from here +0.2 or
distance to a v1 firing spot, standing in the foe's certified shot −0.2,
arrow ETA, dry landing, own resources ready, foe dash/platform on cooldown,
height, distance band. A latch of L decisions costs 0.004·(L−1) (commitment
margin).

**Foe model** (the weak point, every loss mode below traces to it): keeps
its last movement, buttons up (a pending charge fires at once); a charging
foe is scored under both release timings (now / hold to full with tracking
aim) and the candidate keeps the worse; a ready bow shoots once when a
coarse line-of-sight shot exists (charge by range, flat arc tracking me);
with no shot it closes in, and for long charges (8+ decisions) a second
pessimistic model dashes in first; a ready platform within 9 u is planted;
airborne it steers to a landing or spends `bot_recover`; against my arrow it
gets its best dodge in hindsight (min over v1 dodge / dash-away / platform
freeze / none for shot candidates), so a shot only counts as a kill when the
foe cannot escape it — down, mid-air without a dash, on cooldown.

**Cost**: ~4 ms per decision for `search-hard` (≈0.4 s wall per 350-tick game
vs `mixed-hard`; ≈7 s for a full 7200-tick game), ~1 ms for `search-medium`.
A diagnostic `evaluate.py` suite with `search-hard,mixed-hard` (≈400 games)
took 100–160 s wall with two workers.

**Strength (2026-09-11, Rust sim, maps 0–2 both sides).** Bot-vs-bot: 120
games `search-hard` vs `mixed-hard` 87 W / 33 L (72%; `mixed-hard` is
seed-invariant, so the spread comes from the search bot's random candidates;
`search-medium` beat `mixed-hard` in all 6 distinct games and `mixed-medium`
49/59). Against policies via `evaluate.py --games-per-opponent 32` (96
games each; two builds differing only in the platform horizon and a
rejected foe model scored 61–65, 70–81 and 49–58 on the three rows, so read
±8 as noise):

| Policy | `mixed-hard` wins | `search-hard` wins |
| --- | --- | --- |
| overnight-3 trial 001 final (platform squisher, fires no arrows) | 42/96 (44%) | 62/96 (65%) |
| overnight-3 trial 012 final | 24/96 (25%) | 70/96 (73%) |
| run007 final | 41/96 (43%) | 49–58/96 (51–60%) |

Where it still loses: 22 of 33 losses in the 120-game `mixed-hard` set died
*inside an arrow latch*, 19 of them a full charge (hold 16) that the rollout
had scored as a kill — the foe model said the foe would fall or stand, the
real foe recovered or shot first. Against trial 001 all remaining losses are
squishes (32/34). Which of Matthew's points mattered (measured): the
shooting / approaching foe model took the head-to-head from 59% to 67%, the
best-dodge-in-hindsight foe to 72%; platform squish candidates account for
~25% of the search bot's wins (it discovered the squish on its own — 20–36
of its wins in each set are `cause_of_death` 1); the pending own arrow is
handled by the fork for free (zero shoot→dash self-kills observed; a fork
taken after a full charge straight up and stepped through a dash along the
aim kills the shooter on the same tick as the real game does, with the
arrow 6–13 u ahead — `Dash.cs` raycasts the Projectile layer over 13 u); the
cooldown terms in the heuristic were not separately measurable. Later: the
foe model keeps holding a platform it is already growing (the real morph
grows up to 4 s; releasing it at once in the fork hid the squish geometry):
vs trial 001 76→81 of 120, vs mixed-hard 88→89. With neither a shot nor a
firing spot the heuristic falls back to closing in on the foe; without that
the bot camped 16 u from an idle foe until the timeout.

## Game facts the bots use (measured in-sim)

| Fact | Value |
| --- | --- |
| Arrow launch speed | (charge+1)×15 u/s; the hold head reaches charge 0 (holds 1/2/4), 1 (hold 8) and 4 (hold 16) → 15/30/75 u/s |
| Arrow gravity | 40 u/s², no drag; kills on contact (own arrows included — owner immunity is unreliable in practice) |
| Fire delay | ~8–12 ticks after the latch expires (per-hold totals in `BOT_FIRE_DELAY`) |
| Bow | 2.5 s cooldown from morph exit; the bow morph hovers (no fall) and drifts with the aim |
| Dash | 13-unit raycast teleport at the press decision; wall hits attach; exit slide ~7 u; 4 s cooldown; body contact barely pushes (<0.5 u/s), so dash is mobility, not a weapon |
| Platform | morph freezes the body (~72 ticks, hurtbox disabled after ~24), spawns a platform and ejects upward; 6 s cooldown |
| Slime | walk 19 u/s; air accel ~3 u/tick capped ~22; jump launches ~58 u/s decaying by v' = v·(1−1.7/45) − 1.7 per tick (terminal −45); slope jumps launch along the surface normal, not the aim |
| Death | water y < −11.3 (non-space maps); blast zones |x| > 131.25, y > 72.5 |
| Walls | rounded rects with unit axes + corner radius; walking off an edge wraps around the corner and sticks — leaving ground needs a jump or dash |

## Architecture (bopl_bots.h)

Layered decision flow in `bopl_bot_decide`, every layer sharing one `BotView`
(walls, both bodies, cooldowns, all live arrows):

1. **Latch bookkeeping** — while latched, only the aim is live; an arrow
   charge keeps re-solving its ballistic aim every decision until the shot
   actually leaves (`bot_charge_aim`), including through the bow morph.
2. **Dodge** (`bot_dodge`) — earliest predicted arrow impact from trajectory
   simulation (`bot_arrow_eta`); dash away, jump away (never into a gap), or
   platform-freeze (the hurtbox vanishes).
3. **Personality** — archer / pursuer / evasive / mixed (below).
4. **Survival veto** — while airborne, whatever the personality chose must
   still land somewhere dry (`bot_landing_steer`) and not steer into an arrow
   (`bot_steer_hits_arrow`); otherwise try the other steers, then spend a
   recovery resource (`bot_recover`, platform freeze, bow stall).
5. **Launch veto** — a grounded jump or platform ejection rises ~58 u/s and
   can catch an arrow (even the bot's own mortar still climbing away); if any
   horizontal outcome collides (`bot_launch_hits_arrow` at vx −22/0/22,
   because slope jumps launch along the surface normal), the launch is
   cancelled. Separately, with dash and platform both on cooldown a jump is
   vetoed unless every launch outcome can still steer to a dry landing
   (`bot_launch_lands`) — a jump off a low corner near water with both
   recovery resources spent is unrecoverable.
6. **Difficulty noise** — see below.

Shared machinery:

- **Ballistic solver** (`bot_ballistic_arc`, `bot_solve_shot`): flat and high
  arcs per charge, shooter-velocity compensation (morph drift inherits into
  the arrow), an aim-above-center bias for grounded targets, foe motion
  prediction over the fire delay + flight, terrain clearance along the whole
  arc (`bot_shot_clear`), and rejection of solutions whose arrow would
  re-cross the shooter's own post-release free-fall path (`bot_self_cross` —
  an airborne bot firing steeply upward otherwise drops into its own arrow).
- **Firing spots** (`bot_firing_spot`): when no shot certifies from here,
  search all standable wall tops for a spot with a certified arc and navigate
  there; this is what breaks camping stalemates under occluded perches.
- **Navigation** (`bot_nav`, `bot_waypoint`, `bot_surface_y`): walk/jump
  routing with gap checks, dash-climbs gated on the dash actually reaching the
  target (`bot_dash_reaches`), a platform-elevator for high ledges, and
  one-level waypoint search through intermediate perches.
- **Dash safety** (`bot_dash_safe`): a dash must attach to terrain or end
  over dry ground, and its ~55 u/s path must not cross any live arrow.

### Personalities

- **archer** — fires whenever a solution certifies, otherwise walks to a
  certified lip or the foe's deck; does not camp in a range band when no
  shot exists. Platform-elevates only when the foe is nearly overhead.
- **pursuer** — closes distance (waypoints, dash when it lands near the foe,
  platform-elevator), takes point-blank quick arrows and free hits on
  morphing/still targets, snipes falling foes; switches to firing-spot play
  when the foe holds ground a dash cannot reach.
- **evasive** — maximizes distance on a standable perch, shoots only clearly
  safe shots (still/morphing/far targets); a foe still for ≥45 frames
  forfeits the safety rule: the bot walks to a certified spot and takes the
  shot (no stalemates).
- **mixed** — mode switcher with hysteresis: pursuer against falling, morphing
  or passive foes; evasive while the bow reloads at close range with no dash;
  archer otherwise.

### Difficulty knobs (per level: easy / medium / hard)

| Knob | easy | medium | hard |
| --- | --- | --- | --- |
| Dodge window (arrow ETA, ticks) | 10 | 18 | 26 |
| Reaction latency (decisions a threat must persist) | 2 | 1 | 0 |
| Action skip (decision repeats last aim, no buttons) | 22% | 8% | 0% |
| Aim noise (gaussian σ in aim-byte steps, ~1.4°) | 5 | 2 | 0 |
| Full-charge shots under 21 units | no | no | yes |
| Snipes falling foes | no | yes | yes |

Bots draw from their own RNG stream, so a scripted run's map/seed/side
sequences match an unscripted one exactly. Each bot's seed is a pure
function of the episode seed and its player (`bopl_bot_seed`: the episode
seed is `config.seed`, drawn by `bopl_draw_config` and handed to the game by
`bopl_rl_reset`), derived again at every reset and by `research_bot`, so a
game is a function of (seed, map, bots) whatever the environment played
before and whether the bot was selected before or after the reset
(`check_tick_step.py --scripted search-hard` / `mixed-medium` prove it).
Hard v1 bots never draw, so their games did not change when this was fixed;
easy/medium and search games did (same distribution, different games).

## Strength (2026-09-09, run 007 checkpoints, 18 games each = maps 0–2 × 6, both sides, hard)

Bot win rate; timeouts count as neither. `arrows→hits` is the bot's own fire
telemetry over the 18 games.

| Opponent | archer | pursuer | evasive | mixed |
| --- | --- | --- | --- | --- |
| idle (6 games, maps 0–2 both sides) | 6/6, 7→5 | 6/6, 5→5 | 6/6, 7→6 | 6/6, 8→5 |
| random | 94% (17→8) | 89% (14→6) | 83% (4→1) | **94%** (15→4) |
| initial policy | 94% (17→5) | 89% (14→6) | 83% (5→0) | **94%** (17→8) |
| 007 @ 100M | 83% (23→15) | 67% (15→11) | 78% (33→9) | **94%** (29→17) |
| 007 @ 300M | 89% (33→16) | 67% (22→12) | 56% (34→10) | **67%** (20→12) |
| 007 @ 777M (latest) | 78% (29→14) | 72% (18→11) | 33% (20→5) | **56%** (22→10) |

Targets: mixed ≥90% vs random/initial ✓ (94%); the 300M band ask was 30–50%
and mixed lands at 67% — stronger than asked, kept deliberately ("very very
intelligent and hard to beat"). Even 007's latest checkpoint loses to mixed
more often than it wins. `easy`/`medium` dial the strength down.
Policy matchups above were measured before the 2026-09-09 routing fixes
(cling loop, floor-as-cover, unreachable firing spots, drop/climb
flip-flops); the idle gate was re-run after those fixes. Full numbers come
from `evaluate.py --scripted-opponents`. `research/check_scripted_bots.py`
is the cheap idle sweep (maps 0–2, both sides; ~10 s, no GPU).

## Usage

### Evaluation (`research/evaluate.py`)

Scripted opponents are named `kind-level` (`mixed-hard`, `archer-medium`, …)
and are scheduled like `random`/`initial` on the evaluation maps *and* the
held-out maps:

```sh
venv/bin/python native/rl/research/evaluate.py --run <run> --checkpoint <ckpt> --initial <ckpt0> \
    --scripted-opponents mixed-hard,archer-hard,pursuer-hard,evasive-hard
```

Default is `mixed-hard`; empty (`--scripted-opponents ''`) disables. Each
group saves its first game's replay/visual/mind like every other opponent, and
the dashboard labels them "Scripted mixed (hard)". The names are reserved:
a frozen-opponent manifest may not use them (`evaluation_protocol.opponents`).

### ELO ladder (`research/elo_ladder.py`)

```sh
venv/bin/python native/rl/research/elo_ladder.py --run <run> --scripted-anchors mixed-hard
```

Scripted anchors participate in the round robin as `kind='anchor'` entries
(default `mixed-hard`; empty disables). Random stays the 0-Elo anchor.

### Training (`puffer/train.py`)

```sh
venv/bin/python native/rl/puffer/train.py ... --banks 4 --scripted-bots mixed --scripted-level hard
```

`--scripted-bots` (comma list of kinds) permanently converts the **last**
len(list) banks into scripted banks: their environments' opponent player is
driven by the bot in C while the bank's GPU rows still step (fixed graph;
their actions are unread). The learner's share of games against bots is
`len(list)/banks` of the historical half (one bank of four ≈ 1/8 of all
games). PufferLib's self-play pool only manages the remaining policy banks,
so it never swaps weights into a scripted bank; `pool/state.json` stores only
policy banks, which keeps `--resume` exact (scripted banks reload the initial
checkpoint on resume just to keep their unused rows deterministic).
`metrics.jsonl` gains a per-update `scripted` field
(`{"mixed-hard": {bank, games, winrate}}`) and the console line a compact
`scripted` win-rate summary. Both `--scripted-bots` and `--scripted-level`
are resume-checked hyperparameters.

Smoke-verified (10M learner decisions, 8 workers, `--scripted-bots mixed`):
see `native/artifacts/rl/smoke-scripted-001`.

### Standalone (research binding)

```python
from environment import Environment
env = Environment(seed=1, map_id=0, limit=3600, delay=0)
env.reset(2000, map_id=1)
env.bot(1, 4, 2)        # player 1 = mixed (kind 4), hard (level 2); kind 0 detaches
```

`BOPL_BOT_DEBUG=1` prints one line per bot decision (position, mode, aim,
buttons, firing-spot and landing-safety diagnostics) to stderr.

## Pitfalls encountered (why the code looks the way it does)

- **Own arrows kill.** Owner immunity did not reliably protect a bot from its
  own mortar: bots died to their own rising or descending arrows in three
  distinct ways (platform-ejecting into one, dashing at the foe through one,
  jumping off a slope into one). Hence: all arrows are threats; a just-fired
  arrow still adjacent and receding is tagged `leaving` and skipped only by
  the passive sims; dash paths and grounded launches check every arrow.
- **The bow morph hovers.** Firing while airborne then falling into the shot
  is prevented at solve time (`bot_self_cross`), not by a movement rule.
- **Wall extents include the corner radius**, and tilted slabs walk their
  length while pillars walk only their cap (`spanx`); the surface probe
  (`bot_surface_y`) must start outside geometry or it reports phantom perches.
- **Slope jumps launch along the surface normal.** Any jump near arrows is
  vetoed if *any* horizontal outcome collides, not just the aimed one.
- **Cling-jump loops.** Jumping every decision while `rot_side` kept an
  archer on its own spawned platform for two minutes; cling now dashes to the
  perch and rate-limits jumps.
- **The floor is not cover, and neither is a pancake underside.**
  `bot_shot_clear` ignores the support wall only while the arrow is still
  on/above that deck (so a same-platform shot can certify) and rejects blast
  / through-floor shots. Skipping the home wall until near the target used
  to certify shots that stuck in the floor.
- **Firing spots must be reachable.** Unreachable islands made map-1 bots
  idle at a gap until sudden death. Spots are walk/dash/jump reachable, and
  samples include the rounded lip (inboard samples shoot through the deck).
- **Do not treat “almost same x, target below” as arrived.** `dy<3` with
  no `fabs` stood a bot 13 u above the idle foe until the pancake sank.
  Nav to the foe's deck, not the nearest perch-above-me.
- **Threshold flip-flops.** `|dx|<N` walk-direction switches (drop toward
  foe vs nearer lip; climb offset left vs right; `sign(0)=+1`) walked 1 u
  loops for two minutes. Climb/drop pick a stable side.
- **Map-2 islands are round.** `spanx` is ~0 on a circle of radius 6; a
  “tiny stub” test must use `spanx+radius`, or every map-2 platform is
  treated as a stub and dashed off.
- **Sky mortars die.** A 3.7 s 75 u/s high arc peaks above the y=72.5 blast
  zone (`DestroyIfOutsideSceneBounds`). `bot_shot_clear` rejects that path.
