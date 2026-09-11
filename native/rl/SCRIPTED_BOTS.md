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

`search-hard` / `search-medium` / `search-easy` (`bopl_bots_v2.h`) are
model-predictive control: every decision the bot forks the live game
(`bopl_rl_fork`, a private copy with the same random stream), plays a set of
candidate action sequences on the fork for a horizon and emits the first
decision of the best. The levels differ only in the horizon: hard 64 ticks,
medium 48, easy 32 (see *Levels*, below). The bot is deterministic — it
draws no random numbers — so a game against a seed-invariant opponent is
one distinct game per map and side.
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

**Candidates** (≤ 22): stand, walk ±, jump ±, jump up, dash in 4 directions
plus two toward the foe, dash-then-shoot toward the foe (the v1 solver fires
once the dash tail ends), shots for each charge (the v1 solver's aims,
terrain-checked and flat; horizon = charge + fire delay + flight, at least
64), platform (horizon H + 96 covers the ~72-tick morph and ejection). While
latched only the aim varies (an arrow charge searches ±2 steps around
`bot_charge_aim` and the previous aim).

**Scoring**: own death −1 (+0.25·t/H), foe death +1 (−0.25·t/H), both −1;
otherwise a heuristic on the end state: a certified shot from here +0.2 or
distance to a v1 firing spot, standing in the foe's certified shot −0.2
(dropped when the foe is *passive*: neither moved nor morphed for 20
decisions, the v2 bots' `foe_still`), arrow ETA, dry landing, own resources
ready, foe dash/platform on cooldown, height, distance band. A latch of L
decisions costs 0.004·(L−1) (commitment margin).

**Foe model** (the weak point, every loss mode below traces to it): keeps
its last movement, buttons up (a pending charge fires at once); a charging
foe is scored under both release timings (now / hold to full with tracking
aim) and the candidate keeps the worse; a ready bow shoots once when a
coarse line-of-sight shot exists (charge by range, flat arc tracking me);
with no shot it closes in, and for long charges (8+ decisions) a second
pessimistic model dashes in first; a ready platform within 9 u is planted;
airborne it steers to a landing or spends `bot_recover`; against my arrow it
gets its best dodge in hindsight (min over v1 dodge / dash-away for shot
candidates), so a shot only counts as a kill when the foe cannot escape it —
down, mid-air without a dash, on cooldown.

**Cost** (2026-09-11, one core, above a `mixed-v2-hard` vs `mixed-v2-hard`
baseline of 0.23 ms): 0.65 ms per decision for `search-hard`, 0.57 ms medium,
0.53 ms easy (the pre-2026-09-11 `search-hard` cost 3.7 ms). Rating a level
against pool v3 (480 games) takes ~4.5 min with two workers.

**Strength of the first build (2026-09-11 morning, Rust sim, maps 0–2 both
sides; this is the `search-hard` / `search-medium` frozen in pool v3, before
the *Levels* changes below).** Bot-vs-bot: 120 games `search-hard` vs
`mixed-hard` 87 W / 33 L (72%; `mixed-hard` is seed-invariant, so the spread
came from the search bot's then-random candidates; `search-medium` beat
`mixed-hard` in all 6 distinct games and `mixed-medium` 49/59). Against policies via `evaluate.py --games-per-opponent 32` (96
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

### Levels (2026-09-11): why `search-medium` beat `search-hard`, and the fix

Pool v3 rated the first build's `search-medium` above `search-hard` (667 vs
574; joint ladder 699 vs 593; head-to-head 25-22-1). Cause, established by
tracing the class of the winning candidate at every decision and the last
free decision before each death (336 games per variant: maps 0–2, both
sides, seeds 90001–90008, vs `mixed-v2-hard`, `evasive-v2-hard`,
`archer-v2-hard`, `archer-hard`, `random`, run007-final, o3-032-final; sim
`libbopl_sim.so` sha `84568c7b…`):

1. **The six random sequences (hard only) were a defect.** A random
   sequence with an arrow ran to H + charge + 60 ticks (188/220) while every
   other candidate stopped at H = 96, and the end-state heuristic is not
   stationary in the horizon (the foe model shoots once, then its arrow is
   gone and its dash spent), so merely looking further ahead scored higher:
   a random-aim full charge won 12.3% of hard's free decisions (highest mean
   score of any class, +0.445), 69% of those on the heuristic at the long
   horizon, and was the fatal commitment in 58 of 109 losses (61 of the 67
   deaths inside an arrow latch; medium died in a latch 6 times in 55
   losses). Scoring the random sequences at H instead still lost 44 of 87
   games inside them while they were chosen 3.4% of the time: random
   candidates also win the argmax by luck and fail live.
2. **Random aims cost both levels ~100 Elo** (medium 772 → 875, hard 745 →
   795 on pool v3) for the same reason, on a smaller scale.
3. **Hard's extra breadth did not help**: the four extra dash directions (up,
   down-diagonals, straight down) plus the two sideways dash-then-shoot
   candidates cost ~130 Elo (843 → 710; `dashshoot_h` was chosen in 0.04% of
   decisions and preceded 8 of 68 losses); four foe-dodge models instead of
   two ~35 Elo for twice the rollouts; aim perturbations and seven latched
   offsets were neutral.
4. **The horizon has an optimum at 64 ticks** with this foe model (medium's
   candidate set, deterministic, pool v3, two seed bases where run): H=32
   624/613, H=48 828/725, H=64 875/873, H=96 843/814, H=128 632. Hypothesis
   (a) — a longer horizon is a noisier heuristic end state — holds beyond 64.
5. **Determinism exposed camping**: without random candidates the greedy
   search has fixed points (walking into a wall scored 0.184 vs 0.182 for
   the jump for 7000 ticks; 26 of 336 games ran past 4000 ticks, 3 of 6
   idle-gate games timed out). Random exploration candidates (walk / jump /
   dash in a random direction, at H) break them but cost 50–200 Elo (843 →
   790 at H=96; 948–1010 → 725–749 at H=64): chosen 35% of the time, the bot
   wanders. What fixed it is the passivity rule in the heuristic (a foe
   still for 20 decisions is not about to shoot, so no cover penalty):
   idle gate 6/6, and +100 Elo against pool v3, whose opponents camp too.

Nothing time-based is on the decision path (hypothesis d); dash candidates
into arrows were < 0.5% of choices (c).

Resulting build (`bopl_bots_v2.h`, 459 lines, was 462): no random candidates,
no random aims, 4 dash directions, 2 dodge models, 3 latched offsets, the
passivity rule; the levels are the horizon (hard 64, medium 48, easy 32).
Ratings against pool v3 with the `search-*` members excluded (they no longer
describe this code; 480 games each, seed bases 70001 / 80001, cluster-bootstrap
95% CI, sim `84568c7b…` — pool v3 was built on sim `0d4057e7…`, so the
absolute scale differs from `pool.json`):

| build | `search-hard` | `search-medium` | `search-easy` |
| --- | --- | --- | --- |
| pool-v3 build (random sequences, random aims, H 96/64) | 525 [492, 559] | 772 [733, 817] | = medium |
| without the random sequences | 727 [692, 764] | — | — |
| final | **948 [897, 1000] / 1010 [955, 1073]** | 824 [788, 866] / 705 [664, 749] | 655 [618, 694] / 623 [586, 664] |

Same 336-game set, `search-hard`: pool-v3 build 226 W / 109 L / 1 D (mean
694 ticks), final 297 / 37 / 2 (mean 1015 ticks; 48-0 against each of
`archer-hard`, `archer-v2-hard`, `evasive-v2-hard`, 40-8 `mixed-v2-hard`,
31-16 o3-032-final, 37-10 run007-final, 45-3 `random`); the remaining deaths
are 20 squishes and 16 arrows, mostly from the two policies. Deterministic
bots make the CI optimistic (identical games repeat across seeds against
seed-invariant opponents): the two seed bases differ by up to 100 Elo for
medium. Idle gate (`check_scripted_bots.py`): `search-hard` 6/6 (three by
sudden death); medium and easy still stall against an idle foe in 2–3 of 6
games — a deterministic greedy search with a 32–48-tick horizon cannot see
a route off a low ledge. Pool v3's two `search-*` members are stale: a v4
should re-rate them from this code.

## Game facts the bots use (measured in-sim)

| Fact | Value |
| --- | --- |
| Arrow launch speed | (charge+1)×15 u/s; the hold head reaches charge 0 (holds 1/2/4), 1 (hold 8) and 4 (hold 16) → 15/30/75 u/s |
| Arrow gravity | 40 u/s², no drag; kills on contact (own arrows included — owner immunity is unreliable in practice) |
| Fire delay | ~8–12 ticks after the latch expires (per-hold totals in `BOT_FIRE_DELAY`) |
| Bow | 2.5 s cooldown from morph exit; the bow morph hovers (no fall) and drifts with the aim |
| Dash | 13-unit raycast teleport at the press decision; wall hits attach; exit slide ~7 u; 4 s cooldown; body contact barely pushes (<0.5 u/s), so dash is mobility, not a weapon |
| Platform | morph freezes the body (~72 ticks, hurtbox disabled after ~24), spawns a platform and ejects the slime 3.78 u along the platform's *up*, which follows the aim held during the morph (a sideways aim ejects sideways — see v2); 6 s cooldown |
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

## v2 bots (kinds 6–9: `archer-v2`, `pursuer-v2`, `evasive-v2`, `mixed-v2`)

`puffer/bopl_bots_v2_play.h`, dispatched from `bopl_bots_v2.h`; the same
easy/medium/hard knobs. v1 is untouched (kinds 1–4 byte-identical on the
fingerprint). v2 reuses the v1 view, navigation, shot solver and personalities'
shape, and adds layers for the three failure modes the 816-game loss taxonomy
measured on v1 (66% of non-wins were platform squishes, 10% own arrows, ~15%
attacking into a ready bow). Every game fact is cited to the Unity source in
the header comment; the ones that changed the outcome:

- **Own platform morph aims zero.** The aim during the platform morph is the
  platform's controls: Rotation slerps the platform's up toward the aim
  (`PlatformTransform.cs:276-289`) and Exit ejects the slime along that up by
  radius+halfHeight = 3.78 u (`:418-428`; `EjectionOffset` is 0 in the prefab
  data). v1 aims at the foe while morphing, so its platform stands on end and
  throws the slime *sideways* into the nearest wall — that was the real
  mechanism behind 127 of v1's 221 squishes ("in own platform morph at
  death"). A zero aim vector (byte 0, below `DEADZONE`) keeps the platform
  flat; the slime pops straight up onto its own platform, which cannot squish
  its rider. Pressing platform also requires the whole 4.5 u ejection path to
  be clear of other rects (`bot2_platform_press_safe`).
- **Platform threat layer** before the personality: the foe's plant point is
  remembered from its first platforming decision; its spawned rect (hierarchy
  ≥ 300 000 000 on the foe morph's up axis) is tracked once it exists. Within
  3 u (+2 hysteresis, +1.5 once sizing is possible, + closing speed) the bot
  retreats to a standable spot ≥ 11.6 u from the plant preferring a different
  height (sizing reaches ~19 u along one axis, ~11 u diagonally under
  `maxArea`), dashes/jumps when the rect is about to contain it (an airborne
  player cannot be squished), and never presses platform within 11 u of it or
  under a ceiling. Nothing moves *toward* a live plant point: a dash toward it
  is vetoed within 22 u, a walk within 13 u.
- **Real dash safety.** Every dash is re-checked against the 13 u ray Dash
  casts one tick after the press on a mask that includes arrows
  (`Dash.cs:245-258`, `mod.rs:1230-1273`), for every arrow position over the
  next 6 ticks and along the 7 u exit slide; own arrows count once they have
  flown 5 u (`BowTransform.cs:183`). Dodge-dashes are not attempted at eta ≤ 4
  (the press lands next tick and Dash pauses 0.01 s before leaving).
- **Own motion in the arrow ETA.** Grounded velocity is 0 in the state, so v1
  walked under its own returning mortars and uphill lobs. v2 simulates the
  walk the chosen input produces along the ground tangent (`mod.rs:804-812`:
  stick aligned with the tangent ≥ 0.2 runs at 19 u/s along it, mostly
  vertical on a steep slab) and vetoes any grounded walk that meets an arrow
  within 45 ticks, picking the best of right/left/still; the grounded dodge
  evaluates its two runs the same way before jumping.
- **Cooldown/charge-aware engagement.** Foe bow/dash readiness from
  `timers > cooldowns`, its bow aim from the bow hurtbox up vector
  (`mod.rs:2680-2695`). No charge and no closing inside 8.5 u while the foe's
  bow is ready or its bow morph is aimed at me (cos > 0.72 within 15 u); a
  ready bow answers a hold in 12 ticks + range/75 s, so only quick holds
  inside 12 u of one and mid holds inside 20 u; full holds when the foe is
  passive (bow idle ≥ 45 decisions or still ≥ 20) or has its bow down, where
  the hold's charge + fire delay (12/40/72 ticks) + 8 must fit inside the
  foe's remaining cooldown + its quick charge + arrow flight. No duel with a
  foe that is *charging* inside 16 u (a policy re-aims every decision) nor
  with one that released < 0.25 s ago (`foe_bow_fresh`: the bow timer sits at
  0 through the charge and counts from release; its arrow spawns velocity-less
  and fires 8–12 ticks later, invisible to the ETA until then — this was the
  mechanism behind most "killed while morphing bow" deaths). Before any
  charge the arrow ETA is re-run for a *stationary* body (the morph hovers
  where it starts; the dodge layer's ETA assumed the jump/run carried on) and
  the hold is shortened until it fires before that arrow arrives. An airborne
  charge with dash and platform down must also land dry from 6 u lower (the
  morph sinks over a full hold and the exit leaves ~20 ticks without
  abilities). Free-hit rule only for platform morphs from range.
  pursuer-v2/mixed-v2 dash in when the window is open (foe bow down, passive,
  or charging elsewhere) and its dash is down, whenever the 13 u dash ends
  inside mid-hold range (7 < range < 22, |dy| ≤ 7), then shoot from the slide
  (`combo`, 5 decisions); the bow may be entered at any airborne speed
  (`BowTransform.cs:93-97`), the arrow inherits only the residual
  self-imposed velocity.
- **Doomed falls press platform anyway.** The ejection-path safety check is
  skipped when no airborne steer lands (`bot_landing_steer` ×3): a cramped
  ejection risks a squish, the water is certain. evasive-v2 also certifies a
  dry landing (`bot_launch_lands` for the nav steer) before every perch hop.

### Measured (2026-09-11, `research/loss_taxonomy.py`, hard, maps 0–2 × 7 seed pairs × both sides = 42 games per cell)

Checkpoints: `t001`/`t012`/`t002` = overnight-3 trials 001/012/002 finals,
`run007` = `anchors/run007-final.bin`. v1 row = the loss-taxonomy baseline
(same schedule, commit 2539b60), v2 row = kinds 6–9 at this commit.

| bot | t001 | t012 | t002 | run007 | all (wins/168, timeouts) | median game |
|---|---|---|---|---|---|---|
| archer | 0.62 | 0.40 | 0.48 | 0.36 | 0.46 (78, 1 t/o) | 8.0 s |
| **archer-v2** | 0.83 | 0.74 | 0.88 | 0.81 | **0.82** (137, 5 t/o) | 12.4 s |
| pursuer | 0.57 | 0.33 | 0.67 | 0.40 | 0.49 (83, 1 t/o) | 4.5 s |
| **pursuer-v2** | 0.95 | 0.74 | 0.90 | 0.74 | **0.83** (140, 1 t/o) | 11.3 s |
| evasive | 0.74 | 0.52 | 0.71 | 0.50 | 0.62 (104, 5 t/o) | 12.6 s |
| **evasive-v2** | 0.93 | 0.76 | 0.83 | 0.79 | **0.83** (139, 9 t/o) | 13.8 s |
| mixed | 0.50 | 0.40 | 0.45 | 0.33 | 0.42 (71, 1 t/o) | 7.0 s |
| **mixed-v2** | 0.86 | 0.76 | 0.79 | 0.69 | **0.77** (130, 2 t/o) | 12.2 s |

Terminal cause of every bot non-win vs the policies (v1 → v2):

| bot | squished | hit-by-arrow | own-arrow | fell-water | fell-blast-y | timeout | non-wins |
|---|---|---|---|---|---|---|---|
| archer | 63 → **7** | 17 → **14** | 8 → **0** | 1 → **4** | 0 → **1** | 1 → **5** | 90 → **31** |
| pursuer | 43 → **7** | 20 → **15** | 17 → **1** | 2 → **4** | 2 → **0** | 1 → **1** | 85 → **28** |
| evasive | 44 → **3** | 9 → **7** | 3 → **2** | 3 → **8** | 0 → **0** | 5 → **9** | 64 → **29** |
| mixed | 71 → **12** | 16 → **13** | 6 → **4** | 2 → **6** | 1 → **1** | 1 → **2** | 97 → **38** |
| **all** | 221 → **29** (−87%) | 62 → **49** (−21%) | 34 → **7** (−79%) | 8 → **22** | 3 → **2** | 8 → **17** | 336 → **126** |

Sub-buckets that moved: `squished/map-platform` 92 → 11, `foe-platform-growing`
55 → 7, `own-platform-settled` 37 → 6, `own-platform-spawn` 20 → 0 (squished
while in own platform morph 127 → 3); `own-arrow/dash-into-own-arrow` 22 → 0,
`own-mortar-returning` 11 → 1; `hit-by-arrow/dodge-too-slow-jump` 15 → 0,
`killed-while-morphing-bow` 18 → 7 (it was 27–33 before the fresh-release /
stationary-ETA rules above; 24 of those 33 charged into a foe that had just
released or was mid-charge), `point-blank-no-dodge-possible` 14 → 15 and
`dodge-too-slow-steer` 8 → 14 (the bot now stands its ground where it used to
lose the duel — the same short-range encounters, later deaths). Remaining
regressions, left documented: `fell-water/jump` 4 → 18 — from the bot logs
of all 18: 6 are platform-hazard escape jumps from map-2 ledges within 1.5 u
of the water (the airborne platform freeze then spawns too low), 5 are
airborne full holds vs `t002` whose morph sank ~5.6 u and whose exit then
slid off a ledge edge that v1's fall model (`bot_landing_steer`, straight
0.7 u/s² air control, penetration test at the edge) still reported as a
landing, 7 are the same model disagreeing with the real air control near
ledge edges; `timeout/both-active` 3 → 11 (evasive-v2 holding range against
a camping policy, 7 of 11). Median bot death moved from 7.5 s to 11.8 s; bot
wins are still mostly arrows on a grounded policy (380/546). The dash→arrow
combo still fires rarely (8–10 dash→arrow per ~1600 dashes with the loosened
window; the foe-dash-down and open-window conditions seldom coincide).

Head-to-head, hard vs hard, maps 0–2, both sides, 6 seed pairs per cell (the
v1 bots are seed-invariant, so `W-L-T` counts of the row bot over 6 games):

| v2 | vs archer | vs pursuer | vs evasive | vs mixed | total W/L/T |
|---|---|---|---|---|---|
| archer-v2 | 6-0-0 | 4-2-0 | 5-1-0 | 4-2-0 | **19/5/0** |
| pursuer-v2 | 5-1-0 | 5-1-0 | 5-1-0 | 5-1-0 | **20/4/0** |
| evasive-v2 | 5-0-1 | 6-0-0 | 5-0-1 | 5-1-0 | **21/1/2** |
| mixed-v2 | 6-0-0 | 6-0-0 | 5-1-0 | 6-0-0 | **23/1/0** |
| all v2 | 22/1/1 | 21/3/0 | 20/3/1 | 20/4/0 | **83/13/2** |

Against the first build of `search-hard` (kind 5, the pool-v3 build; it
drew random candidates, so with the reset-derived bot seeds every pair was a
distinct game; maps 0–2 × 6 seed pairs × both sides = 36 games per row; the
2026-09-11 *Levels* build sweeps all four v2 kinds 48-0 or 40-8, see above):

| v2 (hard) vs search-hard | W | L | T |
|---|---|---|---|
| archer-v2 | 16 | 20 | 0 |
| pursuer-v2 | 20 | 16 | 0 |
| evasive-v2 | 14 | 22 | 0 |
| mixed-v2 | 20 | 16 | 0 |
| **all** | **70** | **74** | **0** |

(Before the fresh-release/stationary-ETA rules the same protocol gave
62/82/0.) `search-hard` costs ~4 ms per decision against ~30–400 µs here.

Idle gate: all v2 kinds 6/6 on the Unity and Rust backends
(`check_scripted_bots.py`); `check_tick_step.py --scripted mixed-v2-hard`
prints `TICK_STEP_OK`. Decision cost (wall time per `env.step` with two bots
minus the no-bot baseline, loaded machine): v1 archer+mixed ≈ 420
µs/decision, v2 ≈ 415; v1 pursuer+evasive ≈ 21–28 µs, v2 ≈ 34–42 — the
archer's firing-spot search dominates both generations; v2's extra layers
(three `bot_landing_steer` runs per platform press, one arrow ETA per charge)
cost ~10 µs.

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
