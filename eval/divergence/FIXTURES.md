# Reproducible divergences found by the divergence-rate sweep

Not a queue and not a gate: these are cases the unbiased sweep turned up, kept
so that whoever picks one up does not have to rediscover the seed. Each line is
exactly reproducible with the committed tool.

## 1. An active gun-family ability body re-aims in the port and is latched in the oracle

    DR_FIRST=27 DR_GAMES=1 DR_POOL=all DR_TICKS=90 DR_DUMP=78:83:8 \
      artifacts/rl/divergence_rate <oracle.so> <q32.so> <f64.so>

Game index 27, map 27, seed 1703906891, six random-pool ability slots, random
actions. Entity 8 (id 9, kind 1, layer 8, owner 2, source ability 6 Blink Gun,
flags ACTIVE|ENABLED|TRIGGER|CONTACT_DAMAGE_KNOWN).

Ticks 78-80 all three backends agree bit for bit. At tick 81 - an action
decision boundary, where a new aim byte is first applied - BOTH Rust builds
move the body's rotation from 2.7350336 to 0.1478397 and its right/up basis
with it, while the oracle holds the rotation it had at cast. The position stays
bit-identical in the oracle/q32 pair throughout, before and after, so this is
orientation only, and both Rust builds agree with each other, so it is a logic
divergence and not arithmetic.

This is the single largest residual class in the sweep: `entity[].right_x` is
the first divergence in roughly half of all divergent games, and the source
abilities behind it are the aimed-gun family - Bow (4) far in front, then Blink
Gun (6), Growth Ray (8), Shrink Ray (17), Duplicator (29), Black Hole (18).
It is policy-visible: the rows carry the ACTIVE bit, so the observation writer
encodes them.

## Fixture #2 - the port carried entity rows across `bopl_rl_reset`

**Status:** FIXED 2026-09-23 in `d4c1474c` (merged `25743a53`). `leakprobe-q32`
went 338/1240 -> 0/1240 and `DR_FRESH=0` now gives the same first divergences as
`DR_FRESH=1`. Re-confirmed on `706d11a3`: leakprobe 0/1240, selfcheck 0/1240.
Everything below is the original measurement, kept because it is the recipe for
catching a regression and because fixture #3 depended on it.

**Root cause: the rope body pool.** A round restart is
`LoadScene(LoadSceneMode.Single)` (`GameSessionHandler.cs:423`), so `DetPhysics`
is destroyed and its `Awake` rebuilds the pool wholesale as
`ropes = new RopeBodyList()` (`DetPhysics.cs:293`) - 128 fresh `RopeBody` with
`enabled` false - every round, whatever the new loadout holds. `RopeBodyList`
has no reset method and `RopeBody.Dispose` leaves `enabled` set. The port
cleared the pool only when the *incoming* loadout carried Grappling Hook or Web
Swing, which is wrong by construction: what needs clearing depends on the
episode that just ended. Ropes left flagged `physics.enabled` by a rope episode
were published as phantom kind-3 rows into later episodes, because the
observation pass filters on `rope.enabled` alone (`RlObservation.cs:57,:89`).
That is exactly the signature recorded below - kind 3, radius 0.1, parked far
under the map.

When one `BoplRl` handle played several episodes back to back, the Rust port's
exported entity table could gain rows belonging to an *earlier* episode. The
oracle never did this.

Repro (both Rust builds leak; the oracle is the reference):

```
DR_CELL=random DR_POOL=all DR_FIRST=16 DR_GAMES=4 DR_TICKS=3600 DR_FRESH=0 \
  DR_DUMP=1:55 divergence_rate <oracle.so> <q32.so> <f64.so>
```

Game 16 (map 16, seed 1272511178, 1935 ticks, peak 101 entities) is the
poisoning episode. In the *next* episodes the port's entity table is inflated:

| game | map | oracle entities | port entities | extra |
|------|-----|-----------------|---------------|-------|
| 17   | 17  | 13              | 83            | +70   |
| 18   | 18  | 23              | 93            | +70   |
| 19   | 19  | 19              | 89            | +70   |
| 20   | 20  | 16              | 86            | +70   |

Game 19 run on its own is completely clean, so this is history, not the map or
the seed.

Shape of it: the extra rows are *not* present from tick 0. Through tick 40 both
sides hold 18 entities. On tick 41 the oracle spawns one entity and the port's
count jumps 18 -> 89. The 70 extra rows all carry flags 35
(`ACTIVE|ENABLED`, `DESTROYED` clear), `kind 3`, `layer 0`, `radius 0.1`, and
sit at y ~ -40141 - far under the map, where dead objects from the previous
episode were parked.

Two things follow:

1. **They are policy-visible.** `bopl_puffer.h:316` encodes a row when
   `ACTIVE && !DESTROYED`, which these satisfy. A training observation taken
   after a poisoning episode would contain up to ~70 phantom entities.
2. **They do not affect the simulation.** Positions of the real entities stay
   bit-identical (`maxgap` 0), and all three backends end the episode on the
   same tick (111). So this is a defect in the state export, not in physics.

Rate, measured over 1238 random-action episodes played twice - once with
handles reopened per episode, once with handles reused: the reused run reports
a different first divergence in **186 episodes, 15.0% (95% CI 13.1-17.1%)**,
and every one of those 186 is a visible-entity-population jump of this exact
signature. Median inflation +22 phantom rows, maximum +107.

It is bounded, not cumulative: over 80 consecutive episodes the inflation
appears in bursts (games 17-20 at +70, 36-37 at +15, 66 at +22, 69-71 at
+11/+12/+11, 76 at +11) and returns to zero in between. So there is no buffer
exhaustion risk, but roughly one episode in nine is affected when handles are
reused.

**Consequence for measurement.** A first-divergence tick must not depend on
which games happened to run before it in the same process. `DR_FRESH=1` (now
the default) reopens the two Rust handles before each game; the oracle handle
is kept because it does not leak and is the expensive one to load. With it, a
batch of 80 games reproduces 80 fully isolated single-game processes exactly,
across all 16 arm x lens x class combinations, at a cost of 5.2 s for 80 games.

This is also the mechanism behind `replay_ab` returning different answers
depending on how many `.rep` files it is handed - see FIXTURES.md #3.

## Fixture #3 - `replay_ab` results depended on how many replays it is given

**Status:** FIXED 2026-09-23 as a consequence of fixture #2, whose root cause it
shared. The order-sensitivity below should no longer reproduce; the measurement
is kept as the regression recipe.

`eval/rust_ab/replay_ab.c` reuses one handle per backend across every
`.rep` it is handed, so it inherits the cross-episode leak above. The same
replay therefore gets a different verdict depending on its position in the
batch:

```
# alone - reports tick 73, entity[].right_x (twice, deterministically)
replay_ab <oracle.so> <q32.so> random-i000004-map004.rep

# preceded by i000..i003 - reports tick 33, header.entity_count
replay_ab <oracle.so> <q32.so> random-i00000{0,1,2,3,4}-map*.rep
```

`replay_game` does `memset(result, 0, sizeof *result)` (replay_ab.c:458), so
this is not an uninitialised local; the contaminated state lives inside the
port, not in the comparator.

Measured effect: over 40 replays compared one-per-invocation against this
tool, 32/40 agreed exactly on tick and field, 3 more agreed on the tick and
differed only in which field of that tick is reported first, and the remaining
5 were entity-population differences of exactly the fixture #2 kind.

**Consequence, while it lasted.** `make replay-sweep` and the map sweeps were
order-sensitive: adding or removing a `.rep`, or changing glob order, could
change the reported first-divergence tick for unrelated replays. With the rope
pool fixed this no longer holds, but `leakprobe-q32` is the cheap standing check
that it has not come back - it needs no oracle and runs 1240 episodes in 6 s.

## Fixture #1b - the same one-tick aim lag reaches the Bow hurtbox

The pilot attributed about half of all `entity[].right_x` first divergences to
source ability 4 (Bow), which appeared to contradict the C# reading that
`bow.aim` is correctly frozen by `b.released`. Both are right; they are about
different code.

First, the attribution is not evidence that Bow is special: the scripted bots
hardcode Dash/Bow/Platform (`bopl_bots.h:143-149`), so Bow is the only aimed
weapon they ever carry, and in the bot cells *every* `right_x` divergence is
ability 4 by construction (2224 of 2224). Read the bot-cell attribution as
"the aimed-weapon class", not "Bow specifically".

Second, there is a real Bow case, and it is the same class as #1. Repro:

```
DR_CELL=bot:9:2 DR_POOL=fixed DR_SEED=24601 DR_FIRST=3 DR_GAMES=1 \
  DR_TICKS=200 DR_FRESH=1 DR_DUMP=135:139:5 divergence_rate <oracle> <q32> <f64>
```

map 3, seed 1384548501, entity 5 (Bow hurtbox, source ability 4, flags 16395):

| tick | oracle rotation | q32 / f64 rotation |
|------|-----------------|--------------------|
| 135  | 1.9773554       | 1.9773554          |
| 136  | 1.9773554       | 1.9773554          |
| 137  | 1.9773554       | 1.9527154          |

On tick 137 - the hurtbox's last active tick - the oracle holds the aim it had
already latched and the port takes one more update from the live stick. Both
Rust builds move together, so this is logic, not arithmetic; `x`/`y`/`vx`/`vy`
are untouched, so it is orientation-only, exactly like #1.

The mechanism is an ordering difference at `mod.rs:24339-24346`: the port
updates `b.aim` while `!b.released` and only *then* sets `b.released` on the
button-up tick, so the release tick gets one extra aim update that C# does not
take. The exported hurtbox rotation is built from that aim at `mod.rs:31999`.

Not fixed here - recorded for later.

### What triggers fixture #2 (negative results, to save the next person the time)

- **Not one ability.** Every ability 2..38 was played alone (all three slots,
  both players, 8 consecutive episodes, handles reused). None reproduced it.
- **Not simply a big previous episode.** Across 1238 episodes, the episode
  immediately before a contaminated one is barely distinguishable from the
  episode before a clean one: median peak entities 21 vs 20, median length
  269 vs 262 ticks.
- **Bursty.** The 186 contaminated episodes fall into 111 bursts, median length
  1, maximum 10, so poisoning clears on its own.
- **Not reachable from the gate loadout.** With the fixed Dash/Bow/Platform kit
  the existing gates use, the effect never fires: 0 of 1240 episodes
  (95% CI 0-0.31%), against 15.0% with the full ability pool. This is why
  `map-sweep-3000` passes cleanly while reusing one handle across 248 episodes
  (`map_ab.c:479` opens once, `:520` resets per case) - it only ever exercises
  3 of the 38 abilities.

## Fixture #4 - Black hole panicked the port on vector-field maps

**Status:** FIXED 2026-09-23 in `d545562d` (merged `706d11a3`). Kept for the
reproduction recipe, and because the map-27 game listed below is a *different,
still-open* defect that this fix does not touch.

The port aborted with an index-out-of-bounds inside the VectorFieldPlatform
update, always `index out of bounds: the len is N but the index is N`, at
whichever two lines then held:

```
self.platforms[platform_index].active
map.platforms[self.platform_defs[platform_index]]
```

**Root cause.** `self.vector_fields: Vec<usize>` holds *live* platform indices
(filled with `self.platforms.len() - 1` as each authored vector-field platform
is built). `remove_platform` calls `self.platforms.remove(index)`, shifting
every later element down one, and repaired every other live-index holder -
`platform_dynamics`, `platform_defs`, `platform_entities`, `orbits`,
`duplicate_platform_effects`, `spawned`, `players[].platform`, `engines`,
`spike_attacks`, `tesla_nodes`, `macho_boulders`, `scale_changers`, rope
attachments - but never `vector_fields`. Every stored index above the removed
one was then too high by one, and the largest pointed one past the end. The fix
retires entries exactly as `orbits` does. Purely the port's own bookkeeping:
the C# iterates components, so no reference behaviour is involved and no
simulation output changes. `sudden_death_delays` is *authored-def*-indexed, not
live-indexed, so it correctly stays untouched.

**Trigger was Black hole (ability 18).** Played as the only ability in every
slot, 496 episodes each. The seed matters - pin it or the count is not
reproducible:

| loadout | before (`DR_SEED=0x5eed`) | before (`DR_SEED=24601`) | after |
|---|---|---|---|
| Black hole (18) | 13/496 (2.6%) | 11/496 | **0/496** |
| Platform (14)   | 0/496 | - | 0/496 |
| Grenade (3)     | 0/496 | - | 0/496 |

In the unbiased sweep it showed up as 52 of 34,720 attempted games (0.15%),
concentrated on the maps carrying a VectorFieldPlatform: 12, 34, 82, 11, 58,
and also 51.

Crashed games in the unbiased sweep (map, seed, loadout):

```
i1398 map 34 seed 2240342878 [[18,17,16],[3,21,33]]
i2934 map 82 seed 3380414709 [[15,13,6],[18,11,8]]
i3251 map 27 seed 3933741779 [[38,29,4],[10,17,16]]   <- NOT this bug, see below
i3926 map 82 seed  602382264 [[13,18,37],[1,36,28]]
i4994 map 34 seed 1670732068 [[6,10,13],[20,18,6]]
i9188 map 12 seed 3465757549 [[32,11,22],[18,33,1]]
```

All five ability-18 games now play to completion with zero panics.

**The map-27 game is an oracle defect, not a port bug.** An earlier revision of
this fixture guessed at "a second route into the same stale index". That was
wrong, and it is why no single ability reproduced it. i3251 fails on **backend 0
- the oracle** - with a .NET `System.NullReferenceException` in
`ShootDuplicator.ApplyDuplicationEffect` at tick 193. Run the same game with the
port in all three backend slots and it completes cleanly: 274 ticks, terminal,
zero self-divergence. It still fails after the `vector_fields` fix, and no fix
to `vector_fields` could ever have covered it. Open, and owed its own fixture.

**Was unreachable from the gate loadout**, like fixtures #2 and #3: the scripted
bots and the existing sweeps use Dash/Bow/Platform, which contains no Black
hole, so no bot cell crashed once in 13,000+ episodes. That made this the third
distinct defect in this file the fixed-kit gates cannot reach - the argument for
randomising the gate loadouts.

The harness survives a panic regardless: `bopl_rl` catches it, the game is
counted as failed and skipped, and the episode replay is written to /tmp.

### Fixture #2, measured against the library itself

The cleanest measurement of the reset leak needs no oracle at all. Run the q32
build against *two more copies of itself* with `DR_FRESH=1`, which reopens the
two candidate handles per episode but reuses the reference handle:

| configuration | divergent episodes |
|---|---|
| all three handles reused equally (`DR_FRESH=0`), TOL=0 | **0 / 1240** |
| reference handle reused, candidates reopened (`DR_FRESH=1`), TOL=0 | **338 / 1240 (27.3%)** |

Every one of the 338 is `header.entity_count`; no other field ever differs.
Two copies of the same library, given the same seed and the same actions, can
only disagree because one handle had played earlier episodes and the other had
not. That is the leak, with the port/oracle comparison removed entirely, and it
also confirms the simulator itself is deterministic and the comparator sound.

The control script now runs both: `selfcheck-q32` (symmetric, must be 0) and
`leakprobe-q32` (asymmetric, measures the leak).

## The gate that now covers #2, #3 and #4

`make ability-sweep`
(`eval/divergence/run_ability_sweep.sh`). Loadouts drawn uniformly from all
38 abilities, maps assigned by `index % 124` so every authored map is covered
exactly equally, seed pinned at 24601. Four cells, 12,400 distinct episodes
played twice under two handle policies, 43 s wall on four workers:

| cell | games | players | `DR_FRESH` |
|---|---|---|---|
| leak-2p | 9920 | 2 | 1 |
| selfcheck-2p | 9920 | 2 | 0 |
| leak-6p | 2480 | 6 | 1 |
| selfcheck-6p | 2480 | 6 | 0 |

No oracle: the q32 library plays against two more copies of itself, so crashes
and self-disagreement must both be exactly zero. That also keeps the oracle's
own full-pool crash (the map-27 `ShootDuplicator` case above) out of a gate
about the port.

The cell sizes are set by ability x map coverage, measured rather than assumed
- the gate asserts 4588/4588 cells and fails if it ever narrows:

| games | per map | ability x map equipped | ability x map activated |
|---|---|---|---|
| 1240 | 10 | 81.08% | 66.24% |
| 2480 | 20 | 96.16% | 84.87% |
| 4960 | 40 | 99.85% | 93.94% |
| 7440 | 60 | **100%** | 96.36% |
| 9920 | 80 | **100%** | 97.41% |

2p closes the grid at 60 games/map, so 80 is taken for margin. 6p closes it at
20 because it draws 18 slots a game instead of 6. The old `leakprobe-q32` cell
at 1240 games covered only 81% of the grid, which is why it is not the size the
gate uses.

**Measured against both fixed defects.** Each revert rebuilt, both libraries,
all six pre-existing gates re-run:

| build | ability-sweep | the six existing gates |
|---|---|---|
| `72d983d3` (both fixes in) | **OK** crashes=0 divergent=0 | all pass |
| `vector_fields.retain` block removed | **FAIL** crashes 5/9920 (2p), 15/2480 (6p) | all six pass |
| rope-pool fix `d4c1474c` reverted | **FAIL** divergent 2772/9920 (2p), 486/2480 (6p); selfcheck cells 0 | all six pass |

The five 2p crashes are exactly the five *port* crashes listed under fixture #4
- i1398 map 34, i2934 map 82, i3926 map 82, i4994 map 34, i9188 map 12 - and
the gate correctly does not reproduce i3251, which is the oracle's defect. All
36 crashes across the four cells land on maps 11, 12, 34, 51, 58 and 82, the
six carrying a VectorFieldPlatform.

**Not seed luck.** The same broken build re-run at
`ABILITY_SWEEP_SEED=1`, `999983` and `20260923` fails on all three: 2p crashes
6, 15 and 12 against 5 at the pinned seed, 6p crashes 15, 15 and 11 against 15.

**Six players are 12x more sensitive to the vector-field panic** than two:
0.605% of episodes (15/2480) against 0.050% (5/9920). Fixture #4's 0.15%
headline is an average over heterogeneous study cells, not a per-episode rate
that holds at any one player count.

All 2772 rope-pool divergences are `header.entity_count` and nothing else,
across all 124 maps, and the symmetric `selfcheck-*` cells stay at 0 on the
same games - which is what distinguishes a cross-episode state leak from
simulator nondeterminism, and the reason both halves of the pair are kept.

**What this gate cannot see.** A port-vs-oracle *logic* divergence that is
neither a crash nor a self-inconsistency - fixtures #1 and #1b - is invisible
to it by construction. That is what `divergence-rate` is for. And a defect
firing in under ~0.024% of episodes can pass: 12,400 distinct episodes catch a
rate of 0.024% with 95% probability over the choice of pinned seed, 0.05% with
99.8%.

## Fixture #5 - the QuickSpell forms ignored their own boundary component

**Status:** FIXED 2026-09-24 in `a5d3fd9d` (Gust) and `5f905b76` (Revival). All
three gust phases now agree with the reference in every exported field for the
whole episode, and gust has lost its `known_trajectory_divergence` flag, so the
gate protects it. Black hole was still flagged when this was written; it was
fixed later the same day - see fixture #6.

The instrument was `eval/rust_ab/bounds_case_trace.c` (`make bounds-case-trace`),
which replays one `bounds_ability_ab` case in lockstep and reports the first
differing field instead of only the kill:

```
BCT_ENGAGE_Y=-10.9 bounds_case_trace <oracle.so> <q32.so>
  FIRST any tick 367 DISCRETE header.entity_count ref 11 cand 15
```

Not near the kill at all - the *entry* tick, and the same for `early` (348) and
`late` (363). Two causes, both with the arrow pointing from the C# to the port.

**The 0.05 nudge on an airborne cast.** On `early` the first divergence is
`player[0].x`, 0.05 world units apart, which is `PlayerPhysics.UnGround`'s
separation (PlayerPhysics.cs:366-370). `QuickSpell.OnEnterAbility` calls
`physics.SyncPhysicsTo(playerInfo)` *before* `physics.UnGround()`
(QuickSpell.cs:78-83), and `SyncPhysicsTo` assigns `attachedGround` only on its
grounded branch - the airborne branch assigns null (PlayerPhysics.cs:531
vs :549). The ability body is a separate `PlayerPhysics`, so a cast started
mid-air reaches `UnGround` with nothing to push off from and is not displaced.
The port reuses one body and read the slime's own retained `base.ground`, which
survives a walk-off-the-edge fall, so it nudged along a stale normal - at entry
and again at the exit `UnGround`. Revival already went through
`retained_ground`; Gust now does too.

**The form's own DestroyIfOutsideSceneBounds.** Gust.prefab and Revival.prefab
both carry it at hierarchyNumber 4, behind the SlimeController that enters the
ability (Player.prefab, 2) and ahead of their QuickSpell (6) and PlayerBody.
`Updater` sorts a registration batch by that number (Updater.cs:271) and skips
disabled components (Updater.cs:300), so on the entry pass the *slime's*
component at the same 4 is skipped - `onEnterAbility` deactivated it
(SlimeController.cs:190) - and the freshly enabled *form's* component runs
instead, in that same pass. Casting from below the water plane therefore kills
on the entry tick. The port ran the component for the plain slime and for
Rock/Roll/Drill/Platform/Bow but not for either QuickSpell form, so the cast
body - which is frozen for the cast, in both backends - hovered under the plane
for the whole 18-tick cast+exit animation and died only when the slime returned.

Black hole is the same class: `Black hole.prefab` carries the component at
hierarchyNumber 5, also ahead of `BlackHoleClap` (6), and `brink blackhole` is
exactly the entry-pass signature (reference 367, port 368). That prediction held
- and the guess that the two wanted fixing together was right, because the
separate 0.4-0.56 unit x drift did keep the flag on by itself. Both are fixed in
fixture #6, where the drift turned out to be a missing per-pass
`AirealMovement`.

### Negative result worth keeping

`revival` had been a case in `bounds_ability_ab` since the probe was written and
tested nothing. The probe walks off the edge at tick 280 and presses the slot
only once it is off cooldown; Revival's cooldown is 10 s and the body is past the
plane long before then, so the button was never pressed and all three phases
reported a plain unarmed fall that matched trivially. A case can now set its own
walk tick. Measured before the fix: reference 607, port 625 - the identical
18-tick hover. Any case whose slot cooldown outlasts the walk is silently inert;
`revival` was the only one, but the failure mode is invisible in the gate output,
which prints `ok` either way.


## Fixture #6 - Black hole: the cast form never ran AirealMovement, and published its pose a pass out of step

**Status:** FIXED 2026-09-24. All three `blackhole` phases now agree with the
reference in every exported field for the whole episode, and `blackhole` has
lost its `known_trajectory_divergence` flag. **No case in
`bounds_ability_ab.c` carries that flag any more**, so the gate now holds all
38 to the full comparison.

Before, rebuilt from committed source:

```
early blackhole  KNOWN-DIVERGENCE reference 404 (-36.1894,-11.3380) | rust 404 (-36.7478,-11.3380)
late  blackhole  KNOWN-DIVERGENCE reference 391 (-41.0446,-11.3957) | rust 391 (-41.4578,-11.3957)
brink blackhole  KNOWN-DIVERGENCE reference 367 (-42.2811,-11.6208) | rust 368 (-42.2970,-11.6558)
```

After: `ok` on all three, at the reference's own tick and position.

The instrument was again `eval/rust_ab/bounds_case_trace.c`. As with Gust, the
first divergence is the *entry* tick, not the kill - tick 348 on `early`, 363
on `late`, 367 on `brink`, each the tick the button went down. Three defects,
not the two that were expected.

**AirealMovement never ran on the cast form.** This is the 0.4-0.56 unit drift,
and it had not been diagnosed before. Dumping the entry window shows the
reference decelerating horizontally and the port not:

```
reference  x steps after entry: -0.010379 -0.006801 -0.004455 -0.002919 -0.001912
port       x steps after entry: -0.015842 -0.015842 -0.015842 -0.015842 -0.015842
```

The reference's ratio is a constant 0.655172 per pass. `BlackHoleClap.UpdateSim`
ends with `physics.Move(Vec2.zero, simDeltaTime)` (BlackHoleClap.cs:150), and
the airborne branch of `Move` runs `AirealMovement` before its raycasts
(PlayerPhysics.cs:232-240). `AirealMovement` adds the input term and then damps
the horizontal component unconditionally:

```csharp
body.selfImposedVelocity += new Vec2(inputHorz * airAccel, Fix.Zero);
Fix fix = airAccel / (maxSpeed + airAccel);
body.selfImposedVelocity += Vec2.left * fix * body.selfImposedVelocity.x;   // PlayerPhysics.cs:398-403
```

so x keeps `maxSpeed / (maxSpeed + airAccel)` of itself every pass even with
zero input, and with **no delta-time factor** - it is per-pass, not per-second.
The form carries its own `PlayerPhysics`: `Assets/GameObject/Black hole.prefab`
serializes `maxSpeed` 19 and `airAccel` 10, giving exactly 19/29 = 0.655172.
The port's `black_hole_form_motion` applied gravity and integrated but never
called the `air_movement` helper it already had in
`native/src/player_physics.rs`, so the velocity the cast inherits was integrated
undamped and the body kept sliding.

The arithmetic closes exactly. `BlackHoleClap.OnEnterAbility` scales an airborne
cast's velocity by 0.05 (BlackHoleClap.cs:62-66); the slime's pre-entry x
velocity is -19.010123, so the reference's first step is
`0.05 * 19.010123 * (19/29) / 60 = 0.01037925` against a measured 0.010379,
while the port's undamped `0.05 * 19.010123 / 60 = 0.015842` is its measured
constant. The damped series converges after about 0.03 units; the undamped one
never does, which is where 0.4-0.56 comes from by the time the boundary catches
the body.

**The form's own DestroyIfOutsideSceneBounds.** Same class as fixture #5.
`Black hole.prefab` carries the component at hierarchyNumber 5, behind the
SlimeController that enters the ability (Player.prefab, 2) and ahead of its
`BlackHoleClap` (6), so on the entry pass the slime's own component is skipped -
`onEnterAbility` deactivated it (SlimeController.cs:190) - and the form's runs
in that same pass, before the form has moved (Updater.cs:271, 300). The port ran
it for Black hole's steady state but not on entry, so `brink` survived a pass
and died at 368 where the reference dies at 367. Unlike Gust this one did not
need the QuickSpell helper: Black hole stays `active` for the whole cast, so the
ordinary `ability_boundary_kill` marker is converted to a death with cause
`Other` by the shared pass that already handles Beam and Platform.

**The pose was published a pass out of step, in both directions.** Not
predicted, and it survived the first two fixes. `Ability.UpdateSim` publishes
`Player.Position` from hierarchyNumber 2, ahead of `PlayerPhysics` (4),
`BlackHoleClap` (6) and the `PlayerBody` (8) that actually advances the
transform, so the position a pass publishes is the one it *started* with. The
port published after integrating. The hurtbox is the mirror image:
`DPhysicsBox.UpdateSim` is empty (DPhysicsBox.cs:374-376), so its
hierarchyNumber 7 does nothing per pass and the collider follows the body
through `UpdatePhysicsPositions`, which reads `fixTrans.position`
(DPhysicsBox.cs:287-288) and is driven by DetPhysics *after* the pass - so the
exported box sits where `PlayerBody` just integrated to. The port had snapshotted
it *before* the motion. The two rows were therefore swapped relative to the
oracle: the port's player row ran a pass ahead and its entity row a pass behind.
The boundary kill hid this in the gate, because the kill rewrites
`reported_position` and resynchronises both.

### Negative results worth keeping

**Two small residues in the Black hole path are real, pre-existing, and were not
touched.** They are invisible to `bounds-ability-ab`, which compares player 0's
kill, and they are not what this fixture fixed. The second of them turned out
not to be a residue at all - see fixture #8, which closes it - and the lead
chased below (the rounded-rect loop as an unbounded-range channel to a distant
body) was the wrong tree: player 1 moves because the *hole* is 0.082 units out
of place, not because of how its pull reaches player 1:

- A *grounded* cast diverges on the entry tick at `entity[2].right_x`, the
  hurtbox orientation, by 329 raw units (~7.7e-8). Reproduce with
  `BCT_ABILITIES=18,6,29 BCT_ENGAGE_Y=100`. This was confirmed pre-existing by
  building the unmodified base and running the same case: both builds emit the
  byte-identical `cand 5411486426` against `ref 5411486755`. Re-checked with
  `BCT_ALL=1` on 2026-09-24 (fixture #8's instrument): the tick contains exactly
  four differing fields, all of them entity 2's basis, and nothing larger hides
  behind them - so unlike the second residue this one really is small. It is a
  rotation error of order 1e-5 rad, and the field the first-hit walk reports,
  `right_x` at 329 raw, is the *smallest* of the four (`up_x` 26213,
  `right_y` 41287, `up_y` 208).
- On `early` and `late`, a *distant idle* player 1 diverges by ~71 raw units
  (~1.6e-8) on the tick the hole spawns - 16 passes after the cast, `early` 364
  and `late` 379. Attribution is rigorous even though the base build cannot show
  it directly (its own entry-tick divergence masks it): `dr_compare` walks
  header, then players, then slimes, so `header.entity_count`, `player[0].x/y`
  and `player[0].scale` all matching at that tick proves the hole spawned on the
  same pass from a bit-identical position and scale. The difference is therefore
  in the attraction the hole applies to other bodies - `black_holes_tick`, which
  this change never touches.

**A separate, confirmed fixed-point grouping bug in the hole's size math -
found, measured, and deliberately NOT fixed here.** `BlackHole.UpdateSize`
computes the unit first and then scales:

```csharp
Fix fix = startRadius * (Fix)0.01f;          // BlackHole.cs:198
if (t * fix < minSize) { return true; }      // :199  leaves radius UNCHANGED
dCircle.radius = Fix.Max(t * fix, minSize);  // :203
```

The port groups it the other way, `(size * start_radius) * 0.01`
(`native/src/rollout/mod.rs`, in `black_holes_tick`). `Fix::operator*` floors
(`Fix.cs:41447-41464`, mirrored in `fixed_q32.rs:397-399`), and floor-multiply
is **not associative**, so the two disagree. The port's hole is always slightly
too large: +68 raw at spawn size 100, +79 at 115, +89 at 130, +137 at 200 -
the error scales with size. It feeds the influence gate (amplified ~5x by
`influenceRadiusMultiplier`), the eat-contact tests, and the exported entity
radius. The port also clamps unconditionally where C# returns early at
`BlackHole.cs:199-201` and leaves the previous radius in place.

**Fixed separately after this fixture merged**, as its own change with its own
full gate run, because it changes Black hole sizing globally - influence gates
and eat contacts - and is unrelated to the boundary component this fixture is
about. Two things were corrected, not one:

- the grouping, now `let unit = start_radius * Fix::from_f32(0.01);` then
  `size * unit`, matching the C# term for term;
- the clamp. C# returns early below `minSize` and leaves the previous radius in
  place; the port used `.max(min_size)`, which clamps *up*. With the shipped
  constants (`startRadius` 1.24, `minSize` 0.15, `targetSize` >= 100) the
  candidate is about 1.24 and that branch is unreachable, so this half changes
  nothing today and is written faithfully only so it stays right if anything
  ever shrinks a hole.

The +68/+79/+89/+137 figures were reproduced independently from the raw
constants before the fix was applied, by evaluating both groupings under
floor-multiply. They come out exactly, and only when `0.01` truncates to raw
42949672 rather than rounding to 42949673 - which incidentally confirms
`Fix::from_f32` truncates.

It **does not** close the player-1 residue above: the residue's raw values are
byte-identical with and without it (`cand 54678440797`). Whatever moves player 1
is not the radius - correct, and fixture #8 names it: the hole's spawn *position*
read the wrong pass's body position, which is also why `cand 54678440797` never
budged for any change to the radius.

The remaining lead, for whoever picks this up: the rounded-rect loop is the only
*unbounded-range* channel from a hole to a distant body, because the shipped
boulder exception is inverted - `BlackHole.cs:434` guards the influence cutoff
with `|| !dCircle.gameObject.CompareTag("boulder")`, and `BlackHole2.prefab` is
`Untagged`, so every platform in the level is pulled at any distance. The port
reproduces that faithfully, and the force expression, the `GravityForce`
grouping and the `platformMultiplier` placement were all checked term by term
and match. So the next step is runtime, not reading: dump per-tick
`platform_dynamics[i].body.force` and `surface.center` against the oracle's
`roundedRects.physicsBodies[k].force` / `colliders[k].box.center`. Note the port
skips a platform only on `inverse_mass == 0` where C# also skips destroyed and
inactive parents (`BlackHole.cs:426-430`) - an unverified difference.

**The brief's framing was right about the count but not the content.** It
predicted two defects and named the second only as "a 0.4-0.56 unit horizontal
drift, undiagnosed". It also suggested checking whether Black hole's entry path
repeats Gust's stale-`attachedGround` `UnGround` bug. It does not:
`BlackHoleClap.OnEnterAbility` never calls `UnGround` at all (BlackHoleClap.cs:52-76),
and the port's entry arm already clears `ground` without a separation nudge. That
line of inquiry is closed.


## Fixture #7 - reassociated fixed-point products: an audit, and thirteen sites

**Status:** FIXED 2026-09-24 on branch `fix-grouping`. **No gate number moved**,
and that is the headline finding, not an aside - see "Why no gate sees any of
this" below. This entry exists so the audit is not repeated and so the *reason*
the gates are blind to this class is on the record.

`Fix` multiplication is `(a * b) >> 32` with an arithmetic shift, i.e. **floor**
(Fix.cs:41447-41464, `native/src/fixed_q32.rs:397-399`). Floor-multiply is
commutative but **not associative** and **not sign-symmetric**: `mul(a, mul(b,
c))` and `mul(mul(a, b), c)` differ whenever an intermediate product is not a
multiple of 2^32, and `-mul(a, b) != mul(-a, b)` for the same reason. So a
product the C# writes as `a * (b * c)` and the port writes as `(a * b) * c` is a
silent low-bit divergence, and a `Vec2` scaled twice is floored twice per
component where the C# floors a scalar once.

This followed the 2026-09-24 `BlackHole.UpdateSize` fix, which was the first
known member of the class (`startRadius * 0.01` folded first in C#, grouped
`(size * startRadius) * 0.01` in the port; +68 raw at size 100, +137 at 200).

### Method

Every candidate was settled by arithmetic before any code moved: both groupings
evaluated on the **shipped raw constants** under a five-line Python model of
`Fix` multiply and divide. That is both faster and more conclusive than a build -
and note that a build can only ever show this in the **q32** library. The f64
library cannot exhibit a grouping bug at all, because f64 multiply is associative
to within rounding, so a negative f64 result is worth nothing here.

### Correction to the brief, and to `SCOUT.md`

**`Fix` division does not floor.** `Fix.operator/` (Fix.cs:41474) computes
`q = floor(|x| * 2^33 / |y|)` and returns `(q + 1) >> 1` with the sign applied
afterwards: it **rounds the magnitude half-up and is sign-symmetric**.
`checked_div` in `native/src/fixed_q32.rs` matches it. The brief and the shared
scout notes both said division floors too; it does not, and the difference
decides several of the calls below - it is why the magnet fix had to move the
multiplies but not the divides, and why the macho round-trip below is a *rounding*
round trip rather than a flooring one.

### The greps

```bash
# multi-factor products, the brief's starting point
grep -nE "=\s*[a-z_][a-z_0-9.]*\s*\*\s*[a-z_][a-z_0-9.()]*\s*\*\s*" native/src/rollout/mod.rs
# widened: any three-factor chain anywhere under native/src, continuations included
grep -rnE "\*\s*[A-Za-z_][A-Za-z_0-9.:()]*\s*\*" native/src/
grep -rnE "^\s*\*\s" native/src/rollout/mod.rs          # broken-line operand chains
# divisions mixed with multiplications (the a*b/c vs a*(b/c) half of the class)
grep -rnE "\*[^;]*\/|\/[^;]*\*" native/src/ --include=*.rs
# Vec2 scaled more than once
grep -rnE "(complex_mul|normalized_safe|perp|perp_cw)\([^)]*\)\s*\*[^;]*\*" native/src/
```

Everything under `native/src/` was covered, not only `rollout/mod.rs`.
`vector.rs`, `curve.rs`, `physics/mod.rs`, `physics/scene.rs` and `world.rs` were
read in full against their C# counterparts and are clean.

### Fixed

| # | Site | C# authority | Port had | C# has | Error, shipped constants |
|---|------|--------------|----------|--------|--------------------------|
| 1 | `magnet_pull`, 11 expressions | MagnetGun.cs:527,539,546,558,568,572 | `-(direction * pull * strength)` | `((-pullStr) * fix3) * vec2` | 1 raw for the integer pull strengths (sign placement alone), up to 8 raw for `playerPullStr` / `MachoSlimePullStr` |
| 2 | `arrows_integrate` | DetPhysics.cs:1643 | `(force * invMass + UP * g) * physics_dt` | two separately floored terms, and the force term scaled by the **constant** step | 1 raw on ~18% of forces; also dropped the whole impulse while time is stopped |
| 3 | `detonate_smoke` | SmokeGrenadeExplode2.cs:57 | `(offset + noise) * spread * scale` | `Scale * (offset + noise) * spread` | up to 5 raw once `item.scale` is off 1.0 |
| 4 | black hole spawn radius | BlackHole.cs:172-180, DPhysicsCircle.cs:195-204 | registered `startRadius` | `Init` runs `UpdateSize()` **before** `ManualInit()`, so `t * (startRadius * 0.01)` | 188 raw (5325759300 vs 5325759488) |
| 5 | roll substep speed | Roll.cs:259-261 | `speed / n` | `(Fix)1 / n * speed` | differs on 87% of (n, speed) pairs in range, up to 29 raw |
| 6 | roll exit position | Roll.cs:219, 238-241 | `normal * (r_phys - r_player) * scale` | difference of two **already scaled** radii | up to 3 raw off unit scale |
| 7 | meteor smear | MeteorSmash.cs:314 | `off.y * extents.y / y` | `off.y *= extents.y / y` | ~78% of lengths, up to 2 raw |
| 8 | spike relative centre, 2 sites | DPhysicsBox.cs:456, FixTransform.cs:100-103 | `offset.complex_mul(right) * scale` | `FixTransform.offset` is already scaled, so scale goes **inside** | up to 5 raw |
| 9 | rocket engine fire point | RocketEngine.cs:70, FixTransform.cs:73 | `(offset * scale).complex_mul(right)` | `pos + up * (off.y * S) + right * (off.x * S)`, and `up = MulByI(right) = (-r.y, r.x)` | 1 raw on essentially every rotation, **at scale 1 included** - this one is pure sign-asymmetry, not grouping |
| 10 | gust shockwave, 6 target pools | Shockwave.cs:200-213 | `dir * (force / dist) * scale` | `fix3 = force / fix2;` then `fix3 *= scale;` | 1 raw per component pre-impulse, 60 raw after `ForceMode2D.Impulse` |
| 11 | macho boulder throw | MachoThrow2.cs:217, PlayerPhysics.cs:407 (`inverseMass01 = 35` at :67) | `externalVelocity += velocity * forceFromBoulder` | `AddForce(v * ffb / 35)` then `+= force * 35` | up to 17 raw per component - the `/35 * 35` round trip is **not** the identity because the divide rounds and the multiply floors |
| 12 | mine throw position | ThrowItem.cs:213-216 | `side * p.x * scale + up * p.y * scale` | `vec = throwPosition * Scale` first, then `vec.y * up + vec.x * side` | up to 4 raw on the spawn position at off-unit scale |
| 13 | ThrowItem2 throw impulse | ThrowItem2.cs:229 | `direction * scale * force * 60` | `Scale * fix * vec` - scalar first | up to 159 raw per component pre-impulse, 9540 after |

Note that #11 is the same class seen from the other side: the port had *dropped*
a round trip the C# performs. "Copy the C# grouping literally, not the algebra"
cuts both ways.

### Cleared, with the citation

| Site | Verdict |
|------|---------|
| `GravityForce` (BlackHole.cs:278) | `G * fix * mass`, left-associative, port matches. Re-checked, not re-derived. |
| `distance * distance * distance` | C# groups it the same way. Clean. |
| gust platform branch (`* wave.scale` on the **scalar**) | Already the C# form - `fix3 *= scale` before the vector multiply. Not touched. |
| gust player branch, `(scale + 1) / 2` (Shockwave.cs:150-154) | Different quantity (`fix10`), different C#, matches. |
| `local.complex_mul(basis)` in `fire_thrown_item` | `ComplexMul(v, u)` and `ComplexMul(u, v)` are bit-identical - same four products, the two sums merely reordered, and integer addition is commutative. Clean. |
| `dead_zone_dist_squared` | Flagged by a scout as a squared-vs-unsquared mixup. Refuted: `native/tools/extract_rollout_data.py:234` reads `DeadZoneDist` verbatim; the field name is only descriptive. No change. |
| `player_body.rs`, `world.rs`, `curve.rs`, `physics/*.rs` | Swept; no reassociated product found. |

### Why no gate sees any of this

All eight gates were re-run from binaries deleted first (`native/artifacts/rl/`
is gitignored and `make` compares mtimes, so a stale gate binary is the default
failure mode), against both libraries rebuilt from the fixed source. Every number
is **unchanged**: `check_reward: pass`, `ROUTING PASS` x2,
`BOUNDS_ABILITY_GOLDEN_OK cases=38`, `SUDDEN_DEATH_PUBLICATION_OK
maps_mismatched=0`, `pass=211 fail=0 skipped=5`, `compared_snapshots=417009`,
`compared_snapshots=801501`, `ABILITY_SWEEP_OK episodes=24800 ... divergent=0`.

That is expected, and it is structural:

- **`ability-sweep` has no oracle.** It runs the port against two more copies of
  itself (leak / selfcheck cells). It can catch a crash or self-disagreement. It
  can never catch a grouping bug, because all three copies compute the same wrong
  product.
- **Every other gate plays the hardcoded Dash/Bow/Platform kit**
  (`train/bopl_bots.h:143`). Magnet (30), Roll (16), Macho, Mine, Smoke, Spike,
  Rocket engine, Meteor and Black hole are all outside it. Ten of the thirteen
  sites are unreachable from the gate loadout at all.
- **`dr_compare` reports only the first differing field**, walking header ->
  players -> slimes -> entities, and within an entity row `right_x` comes before
  `radius` (`eval/divergence/divergence_compare.h`). An entity-radius divergence
  is invisible unless nothing earlier differs.

So the whole class is argued from the C# and from exact arithmetic, and is *not*
measured by the suite. That is the honest status.

### Negative results

**The brief's black-hole spawn-radius lead is real in the C# but not observable
in the export.** Finding #4 above is genuine - `Init` does register
`t * (startRadius * 0.01)`, 188 raw below `startRadius` - but the exported row is
not where it shows. A zero-tolerance scratch probe driving the oracle and the q32
port through `black_hole_ability_ab`'s action script and printing the layer-23
entity radius every tick measured **delta 0 at every tick, including the first
published one (199)**: the hole's time-zero `UpdateSim` runs later in the same
updater pass, so the published radius is already the post-tick value in both
backends. (`black_hole_ability_ab.c` uses `SPATIAL_TOLERANCE 70000000`, ~0.0163
world units, so it could not have caught 188 raw in any case.) The real latent
window is narrower and has nothing to do with the export: `magnet_pull` and
`magnet_try_pickup_item` read `hole.radius` from **inside the player loop**,
ahead of `black_holes_tick` (`native/src/rollout/mod.rs:5240`), so a Magnet
player ordered after the caster queried the wrong circle on the cast tick. Fixed
anyway, because the value was wrong.

**A clean "hole spawns and survives with nothing earlier diverging" case is not
constructible with the instruments that exist.** `bounds_case_trace` with
`BCT_ABILITIES=18,6,29` holds the ability button for the rest of the run, so the
hole is never released and never survives; and at the cast tick it already
reports a pre-existing 329-raw `entity[2].right_x`, which is fixture #1's aimed-
gun class and is compared *before* `radius` regardless. Turning this class from
argued into measured needs one of: a comparator that reports **all** differing
fields rather than the first (or a per-field mask), a probe that can release a
cast and then idle, or an ability-sweep variant with an oracle. The comparator
change is the cheapest and would pay for itself across fixtures #1, #6 and this
one.

**The comparator change was made** (`dr_report_all` in
`eval/divergence/divergence_compare.h`, exposed as `BCT_ALL=1` in
`bounds_case_trace`; `BCT_IGNORE` exposes the mask and `BCT_DUMP_ALL=1` widens
the per-tick dump past owned rows). It found a real defect on the first case it
was pointed at - see fixture #8 - so this entry's two open residues were not
residue. It is NULL by default, so `divergence_rate` still pays no formatting
cost on a clean tick.

**`divergence_rate` is too noisy to measure any of these.** A pre-existing,
loadout-independent `header.entity_count` divergence fires at ticks 57-112 on ~5%
of maps, and an `entity[].owner` divergence at median tick ~153-273 on every
pool, masking everything downstream. Confirmed loadout-independent by running
`DR_POOL=30` against `DR_POOL=2` at `DR_TICKS=56` and `DR_TICKS=120`: identical
0/124 and 6/124 counts. Those two are open defects in their own right and are not
part of this audit. The 114-case sweep below reports `header.entity_count` on 13
bounds cases too, but on the death tick and only for particular kits, so treat it
as a separate defect with the same field name until something measures the two
together - see class 5 there.

## Fixture #8 - Black hole spawned from the wrong pass's body position

Found 2026-09-24 by the first case the new report-all comparator mode was
pointed at, which is the whole argument for having built it.

**The symptom, as it had always read.** `bounds_case_trace` on the Black hole
kit reported one differing field on the `early` phase:

```
FIRST any tick 364 NUMERIC slime[1].x ref 54678440726 cand 54678440797
```

71 raw on the *opponent's* slime. That reads as last-bit rounding residue, and
fixture #6 recorded it as exactly that: a small pre-existing residue, left open.

**What the tick actually contained.** `BCT_ALL=1` prints every differing field
of the first diverging tick instead of the first one the accumulator keeps. The
tick is 28 fields over slime 1 and entities 0 and 10-15, and ranked by raw gap
the picture is not subtle:

```
entity[0].y   gap 352392313 (0.082048)   <- the hole
entity[14].vx gap   1352507 (0.000315)
entity[14].vy gap    512558 (0.000119)
entity[11].vy gap    379952 (0.000088)
...
slime[1].x    gap        71 (0.000000)   <- the field that was being reported
```

Four orders of magnitude between the cause and the field the first-hit walk
happened to reach first. `dr_compare` compares header, then players, then
slimes, then entities, so the 71-raw slime always won.

**What entity 0 is.** Row 0 is a reused pool slot: it holds the water (id 1,
layer 4) through tick 363 and at 364 holds a new row, id 16, kind 0, layer 23 -
`RigidBodyAffector` in `ProjectSettings/TagManager.asset` - radius 1.263317
growing to 1.284967 the next tick. That is the Black hole itself, appearing in
both backends on the same tick with the same radius, so the spawn *timing* was
never wrong.

**Root cause.** `BlackHoleClap.FireCommon` (BlackHoleClap.cs:85-91) places it at
`body.position + Fix.Clamp(body.fixtrans.Scale / 3, 1, 2) * ComplexMul(body.right, blackholeOffset)`,
with `blackholeOffset` raw 25769803776 / 12884901888 = exactly (6, 3)
(`Assets/GameObject/Black hole.prefab:236`). Subtracting the offset from the
published hole position gives each backend's `body.position` at the spawn:

```
reference  (-36.293827, -1.458210)   = the caster's position at tick 363
port       (-36.293845, -1.540258)   = the caster's position at tick 364
```

One pass apart, and the gap is exactly that pass's fall: the caster's per-tick
y displacement is 0.078784 at 363, **0.082048 at 364**, 0.085282 at 365. The
`late` phase gap, 0.085823, is its own tick's displacement. Not rounding at all.

The ordering is on the prefab. Resolving every `hierarchyNumber` on
`Black hole.prefab` against the script guids gives 1 SpriteAnimator, 2 Ability,
3 PlayerCollision, 4 PlayerPhysics, 5 DestroyIfOutsideSceneBounds,
6 BlackHoleClap, 7 DPhysicsBox, **8 PlayerBody**. The spawn runs from an
animator callback (`beginAnimThenDoAction(... FireAir)`), and the form's own
`PlayerBody` - which integrates the fall - is the *last* component in the pass.
The slime's own components never compete for the ordering: entering the cast
deactivates that GameObject and `Updater` skips disabled components
(Updater.cs:300).

**The fix** (`native/src/rollout/mod.rs`) keeps one extra Vec2 on
`BlackHoleCastState`, `spawn_origin`, shifted at the top of every cast tick
before anything can spawn from it, and `spawn_black_hole` takes it as an
argument instead of reading `slot.base.position`. All three phases then report
`FIRST any none / FIRST discrete none` for the whole episode, where before the
fix `early` and `late` diverged at 364 and 379.

**Honest limit, recorded because the next person will ask.** The offset needed
is one pass older than the position this port has integrated to when its cast
tick runs - not the pre-motion position of the current pass, which is what
`base.position` already is there (`black_hole_form_motion` runs at the *end* of
`black_hole_cast_tick`, and nothing moves the body earlier in the pass). Tried
and measured as no-ops first: `slot.reported_position` and a one-deep
`pre_pass_position` stored after the motion, both of which equal
`base.position` at that point. Only the two-deep shift matches, and it matches
exactly. So *what* the correct origin is has been measured on three phases; the
remaining question is why this port's form motion sits one pass ahead of the
oracle's for this ability, which needs the Updater registration order for a
`PlayerBody` enabled mid-cast. Whoever picks that up should check whether other
ability forms that spawn something from an animator callback have the same
one-pass skew, since the fix here is local to Black hole.

## Fixture #10 - what a death publishes: the rope's detach bit, and four defects stacked behind it

Numbered after #9 but written in parallel with it and merged second, so it sits
earlier in this file than the fixture it follows. The sections here are in the
order they were appended, not in numeric order.

Found 2026-09-24 by taking class 5 of the 114-case sweep - "a different number
of objects released by the death", thirteen cases - and driving each one to
`FIRST any none`. Eleven of the thirteen closed. They needed **five** separate
fixes, not the one or two the class description implied, and each fix only
became visible once the one before it was out of the way. That stacking is the
result worth keeping: a class in that sweep is a symptom bucket, not a defect.

### The starting measurement

`bounds_case_trace` with `BCT_ALL=1`, the five kits of class 5 x three engage
heights, on `847d7c9b`:

```
CASE            FIRST any (before)                                    AFTER
grapple-early   tick 348 NUMERIC entity[0].x  -36.263780/-35.763780   none
grapple-late    tick 363 NUMERIC entity[0].x  -41.014505/-40.514505   none
grapple-brink   tick 367 NUMERIC entity[0].x  -42.281136/-41.781136   none
beam-early      none                                                  none
beam-late       none                                                  none
beam-brink      tick 367 DISCRETE header.entity_count  ref 10/cand 9  none
roll-early      tick 371 DISCRETE header.entity_count  ref 11/cand 12 none
roll-late       tick 367 DISCRETE header.entity_count  ref 11/cand 12 none
roll-brink      tick 367 DISCRETE header.entity_count  ref 11/cand 12 none
grenade-brink   tick 367 DISCRETE header.entity_count  ref 11/cand 12 none
smoke-brink     tick 367 DISCRETE header.entity_count  ref 11/cand 12 none
grenade-early   tick 465 DISCRETE header.entity_count  ref 12/cand 13 class 1
grenade-late    tick 393 DISCRETE header.entity_count  ref 12/cand 13 class 1
smoke-early     tick 465 DISCRETE header.entity_count  ref 12/cand 13 class 1
smoke-late      tick 393 DISCRETE header.entity_count  ref 12/cand 13 class 1
```

**The sweep's own record of grapple is wrong and this is why.** It lists
grapple's first divergence as 383 / 378 / 367 - the death ticks - and describes
only the rope-flag tail. At `BCT_AIM=0`, which is what the reproduce recipe in
that section prints, grapple first diverges at the *engage* tick, 348 / 363 /
367, by exactly half a world unit on the hook. The sweep's numbers are the
`FIRST discrete` line, or were taken at the case table's `aim 64` where that
defect does not fire. Both aims are now clean; the run above is `BCT_AIM=0` and
an `aim 64` repeat gives the identical 11-clean / 4-class-1 split.

### 1. The rope is never marked detached when its owner dies

The one the class description named. `flags` 35 in the reference and 3 in the
port on every rope node, `kind 3`, `layer 0`, one bit: 32.

`PlayerCollision.killPlayer` detaches the slime's rope as its **first** action,
ahead of the invulnerability early-return, so even a `killPlayer` that returns
false detaches (PlayerCollision.cs:106-114). `PlayerBody.AttachRope(null, top)`
(PlayerBody.cs:253-273) runs `SetForce`, `SetTimeInAir((Fix)100)` and then
`RopeBody.Dettach`, which is where `hasBeenDettached` is set (RopeBody.cs:90-92)
and what `RlObservation.cs:89` publishes as entity flag 32. `Rollout::kill`
touched the rope nowhere; `E_DETACHED` had no writer on any death path.

The fix is four lines at the top of `kill`, before the poke-invulnerability
return, calling the port's existing `attach_player_rope(index, None, top)` -
which is `PlayerBody.AttachRope` field for field.

**Which call sites.** `Rollout::kill` models both `killPlayer` and
`Ability.ExitAbility`, and only the first detaches: `Ability.ExitAbility` hands
the rope to the exit info and calls `EnterAbilityClearRope`, which nulls
`ropeBody` without ever detaching (Ability.cs:139-146, PlayerBody.cs:275-280).
An unconditional detach would therefore be wrong, as the brief warned. The
separator is `active == ActiveAbility::None`, and it is exact rather than
approximate: a player inside an ability form had `PlayerBody.ropeBody` nulled on
entry, so `killPlayer`'s detach is a no-op for them anyway, while the rope such
a form still holds belongs to `BounceBall`, whose own `AttachRope` contains no
`Dettach` call at all (BounceBall.cs:205-223). The `is_some()` test is not a
guard either - `killPlayer` only calls `AttachRope` when `ropeBody != null`, and
the `prev*` velocity snapshot inside it is conditional on that in the C# too.

**Deliberately not reproduced.** `AttachRope`'s `SetForce` honours
`ignoreGravityWhenRoped`; `attach_player_rope` hardcodes rope gravity. That
predates this work, is not observable in any of the thirteen cases, and was left
alone.

### 2 and 3. Two older grapple defects the rope tail was sitting on

Neither is a death-tick defect. Both were invisible while the case diverged
earlier.

**The hook spawns half a unit off with no aim input.** `HookshotInstant.FireHook`
computes `fixTrans.position + player.AimVector() * hookSpawnOffset` and
substitutes `Vec2.right` only in the *velocity* argument
(`((player.AimVector() == Vec2.zero) ? Vec2.right : player.AimVector()) * hookSpeed`,
HookshotInstant.cs:152-156). The port folded the fallback in before computing
either, so with no aim it spawned the hook, and with it the whole rope's
initialisation, 0.5 to the right. The oracle at the engage tick puts the hook
exactly on the body and the rope in a degenerate vertical column; the port had
it already spread along +x. Visible from the engage tick to the end of the
episode, on all three phases.

**The hook clears activeInHierarchy on hits that do not destroy it.** `flags` 13
in the reference and 12 in the port on the tick the hook lands. Bit 1 of a
collider row is `transform.gameObject.activeInHierarchy` (RlObservation.cs:115).
`RopeHook.OnCollide` ends in `Updater.DestroyFix(hitbox)` and
`Updater.DestroyFix(this)` - the hitbox and the component, not the object
(RopeHook.cs:182-189) - so the bit survives a hit on a wall, terrain, an orb or
a black hole.

**And this one immediately over-corrected, which `dash_grapple_form_ab` caught.**
Applying "keep the bit" to all three of the port's collide branches flipped the
same field the other way: reference 12, port 13, on a hook that hit a slime. The
Player branch is the single exception - when the victim has a `PlayerBody` it
destroys the whole hook GameObject and `return`s before the shared teardown
(RopeHook.cs:144-150). Three branches keep the bit, one does not. The probe
existed and found it in the first suite run, which is the argument for running
the whole suite rather than the case you are working on.

### 4. Only the selected throw hurtbox was freed on death

This is the entity-count half, for `roll`, `grenade` and `smoke` - the
`{3, 13, 16}` kit in all three slot orders.

The count is a row-pool count, so reading it needs the partitions. At the death
tick of `grenade-late` the reference publishes 12 rows and the port 13, and the
extra one is a box: seven boxes before the death, four after in the reference
and five in the port. Resolved by id, the reference frees player 1's slime box,
its Grenade hurtbox **and its Smoke hurtbox**; the port frees the first two.

Every ThrowItem2 slot owns a hurtbox - `throw_entities[ability_slot]` - and
`throw_entity` is only whichever one is currently selected, reassigned on every
ability switch. All three of `kill`'s removal lists used `throw_entity`. The
next-pass cleanup for `out` players, thirty lines away in the same file, already
listed `throw_entities[0..2]` correctly, which is why the divergence appears on
the death tick and is gone by the next one.

The 76-to-82 "differing fields" these cases reported were one displaced row
counted once per field of everything after it, exactly as the class description
predicted.

**This is not fixture #7's `header.entity_count` defect.** That one fires at
ticks 57-112 on ~5% of maps and measured loadout-independent. This is a
per-player ability hurtbox not freed for one pass, fires only on the death tick,
and only for kits holding more than one ThrowItem2 ability. Different mechanism,
same field. Fixture #7's own case was not re-measured here.

### 5. Beam's orb, spawned after its own bounds component and destroyed the pass it was born

`beam-brink`, the one case in the class where the reference has *more* rows than
the port: ref 10, cand 9. The missing row is `kind 0`, `layer 18` (Explosion),
radius 0.96, at `body.position + staffDir * 2.8` - the beam orb, which
`beam-late` shows both backends spawning normally at their engage tick.

`brink` is the phase where the engage tick and the death tick are the same tick,
and two orderings were wrong at once.

`Beam.OnEnterAbility` instantiates the orb itself (Beam.cs:261-267), during the
ability entry that `SlimeController.OldUpdate` performs, before `Updater` has
reached any updatable of this pass - including the form's own
`DestroyIfOutsideSceneBounds`, which is hierarchyNumber 4 on `Beam.prefab`
(1 Ability, 2 SpriteAnimator, 3 PlayerPhysics, 4 DestroyIfOutsideSceneBounds,
5 PlayerCollision, 6 DPhysicsBox, 7 PlayerBody, 8 Beam). The port called
`ability_boundary_kill` first and returned, so the orb was never created.

Moving the spawn ahead of the kill was not enough on its own, and the second
half is the more interesting one. `Beam.OnEnterAbility` also calls
`DisableIfNotHeld_deter.HoldThisFrame` on the orb (Beam.cs:270). That component
destroys its GameObject only when `holder == null` and deactivates it only when
`!isHeld`, resetting `isHeld` at the end of each pass
(DisableIfNotHeld_deter.cs:22-47). Held on the pass it was born, the orb cannot
go on that pass - it goes on the next one. The port's `beam_orbs_tick` removed
it as soon as the owner stopped being in Beam, with no born-tick test on that
branch, so the orb the kill had just created was removed in the same pass it was
created. Both had to change.

### What is left, and it is not this defect

`grenade` and `smoke` on `early` and `late` still diverge, and after the count is
fixed the residue is two fields:

```
grenade-late  entity[1].x gap 136006159 (0.031666)   .y gap 348763201 (0.081203)
grenade-early entity[1].x gap 136079760 (0.031684)   .y gap 545057382 (0.126906)
```

Those numbers are **byte-identical** to the ones this file already records for
`mjolnir`, `poke_ball` and `tack` under class 1 of the sweep - the ThrowItem2
payload released when its holder dies, read from a body position one pass too
late. Same defect, arrived at from a different kit and a different class. Class
1 and this half of class 5 are one defect, not two, and the count of "cases in
class 1" should be ten rather than six. Not fixed here: that code was being
worked on concurrently and was left untouched deliberately.

### Negative results

- **Defect A is not upstream of the count defect.** Measured, not assumed: with
  the rope detach landed and nothing else, `beam-brink`, `roll-early`,
  `grenade-late` and `smoke-brink` report first-divergence lines *identical* to
  the baseline, character for character. `grapple` and `beam` being the same kit
  in different slot orders is a coincidence of the case table.
- **There is no missing `E_ENABLED` on the Smoke hurtbox.** Reading the shifted
  rows of `grenade-late` suggested the reference had flag 2 where the port did
  not. That was the row shift comparing a Grenade box against a Smoke box. Once
  the counts align, every flag field matches.
- **The half-unit hook offset does not fire at `aim 64`**, which is what the
  bounds-ability case table uses, and is why the gate never saw it and why the
  sweep's grapple row disagrees with a `BCT_AIM=0` run.

### Reproduce

```bash
BCT_ABILITIES=27,28,9 BCT_SLOT=0 BCT_AIM=0 BCT_ENGAGE_Y=0.0 BCT_ALL=1 \
  native/artifacts/rl/bounds_case_trace \
  "$PWD/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so" \
  "$PWD/native/target-q32/release/libbopl_sim.so"
```

Kits: `grapple` 27,28,9 - `beam` 28,27,9 - `roll` 16,3,13 - `grenade` 3,16,13 -
`smoke` 13,16,3, at `BCT_ENGAGE_Y` 0.0, -8.0 and -10.9. `BCT_DUMP=a:b` with
`BCT_DUMP_ALL=1` is how the row partitions above were read; note that the dump
prints only rows carrying `E_ACTIVE`, so a freed pool slot shows up as a gap in
the `e<N>` numbering rather than as a line.

### Gates

All eight, each from a deleted-and-rebuilt binary with both libraries rebuilt
(`cargo build --release --manifest-path native/Cargo.toml` and `make rust-q32`),
verbatim:

```
########## GATE check-reward ##########
check_reward: pass
########## GATE check-routing ##########
ROUTING PASS .../native/artifacts/rl/check-routing-oracle/result.json
ROUTING PASS .../native/artifacts/rl/check-routing-port/result.json
########## GATE bounds-ability-ab ##########
BOUNDS_ABILITY_GOLDEN_OK cases=38 water=-11.3 survivors=4
########## GATE sudden-death-publication-ab ##########
SUDDEN_DEATH_PUBLICATION_OK maps_mismatched=0
########## GATE rust-ab-suite ##########
pass=211 fail=0 skipped=5
########## GATE map-sweep-3000 ##########
MAP_STATE_OK maps=124 player_cases=2 compared_snapshots=417009 nonexact_map_rows=0 max_vector_raw=0 max_basis_raw=0 max_scalar_raw=0
########## GATE map-sweep-endgame ##########
MAP_STATE_OK maps=124 player_cases=2 compared_snapshots=801501 nonexact_map_rows=0 max_vector_raw=0 max_basis_raw=0 max_scalar_raw=0
########## GATE ability-sweep ##########
ABILITY_SWEEP_OK episodes=24800 maps=124 abilities=37 ability_map_cells=4588 crashes=0 divergent=0 seed=24601
```

`bounds-ability-ab` printed 114 phase rows with zero `KNOWN-DIVERGENCE`. Every
number is identical to the pre-change run: the rope detach, the beam-orb
reordering and the hurtbox free are all changes that could have moved
`ability-sweep` or either map sweep, and none of them did. The fifth gate needed
two attempts - the first `rust-ab-suite` failed `dash_grapple_form_ab` at tick
164 on `flags 12/13`, which is the hook-`activeInHierarchy` mistake described
above; the numbers here are from the run after that was corrected.

## The 114-case trajectory sweep - what bounds-ability-ab never looked at

`bounds-ability-ab` compares the boundary kill and nothing else. Fixture #8 was
half a world unit of horizontal drift and it sat behind that gate for the whole
project, because the kill tick and the kill y still matched. So the gate passing
says nothing about the 37 abilities whose trajectories it never reads.

On 2026-09-24, on the build carrying the fixture #8 fix, I ran
`bounds_case_trace` with `BCT_ALL=1` over all 38 cases x 3 engage heights and
read every differing field of the first diverging tick. **84 of 114 clean, 30
divergent** - and then the 30 turned out to be far better news than the count
suggests.

### The one fact worth taking away

**Twenty of the thirty diverge on the death tick itself, and nowhere earlier.**

```
CASE               FIRST DIVERGENCE   DEATH (reference / candidate)
roll-early         371                371 / 371
roll-late          367                367 / 367
roll-brink         367                367 / 367
smoke-early        465                465 / 465
smoke-late         393                393 / 393
smoke-brink        367                367 / 367
grenade-early      465                465 / 465
grenade-late       393                393 / 393
grenade-brink      367                367 / 367
beam-brink         367                367 / 367
grapple-early      383                383 / 383
grapple-late       378                378 / 378
grapple-brink      367                367 / 367
                   ^ wrong: see fixture #10. These three are the `FIRST discrete`
                     line. At BCT_AIM=0 grapple first diverges at the engage
                     tick, 348 / 363 / 367, on the hook's position.
missile-early      371                371 / 371
mjolnir-early      465                465 / 465
mjolnir-late       393                393 / 393
poke_ball-early    465                465 / 465
poke_ball-late     393                393 / 393
tack-early         465                465 / 465
tack-late          393                393 / 393
```

Every one of those twenty runs bit-exact for 367 to 465 ticks, kills at the
identical tick, and then disagrees about what the death *released*. The
trajectory is not the problem; the port's fidelity through the flight is better
than the raw 30/114 reads. What is wrong is one event surface - the objects a
dying player drops, and the pass whose state they are built from.

Of the remaining ten: three are a spawn-hierarchy counter at tick 0, two are
`platform` on cases where nobody dies, two are a Rewind timer, and three are
last-bit noise.

This is also why `bounds-ability-ab` is green on all of it. That gate reads the
death tick and the death position, both of which match in all twenty; the
divergence is in the objects published on the same tick.

Per-case logs land in `/tmp/bct-sweep/<case>-<phase>.txt`, which is scratch and
not committed; the reproduce recipe is at the end of this section.

### 1. The dying player's released ability object spawns a pass late - 6 cases

**Status:** FIXED 2026-09-24, see fixture #9. All six now report `FIRST any
none / FIRST discrete none`, `tack`'s `entity[].owner` included. The
conclusion drawn at the end of this section - that the port integrates the
body a pass ahead of the oracle, and that fixture #8 and this share one
underlying offset - **was wrong**; it was an ordinary in-pass ordering error
on the ThrowItem2 prefabs. Everything below is the original measurement.

`mjolnir`, `poke_ball`, `tack`, each on `early` and `late`. The diverging row is
a layer-10 (item) entity, `kind 0`, `flags 2307`, appearing on the death tick
itself, and the gaps are **byte-identical across all three kits**:

```
early   entity[].x gap 136079760 (0.031684)   entity[].y gap 545057382 (0.126906)
late    entity[].x gap 136006159 (0.031666)   entity[].y gap 348763201 (0.081203)
```

I first read the identical gaps as evidence of a single shared scene item, since
Mjolnir is `{37, 32, 36}` and shares no ability with Poke ball `{33, 2, 14}` or
Tack `{35, 2, 14}`. That was wrong, and `BCT_DUMP_ALL=1` says so: the object is
kit-specific, radius 0.55 for Mjolnir, 0.93 for Poke ball, 0.50 for Tack, with
Tack alone carrying `rot 0.066667`. It is each ability's own object, released
when the ability exits because its holder died. The gaps agree only because all
three kits die at the same tick, so "one tick of that player's motion" is the
same number in each - which is the useful part of the finding, not the
coincidence.

The spawn position is the one that is stale. At `early`, player 0 is at
(-39.939071, -11.182124) on tick 464 and (-39.970754, -11.309030) on 465, the
tick it dies. Every release offset is a fixed vector from a body position -
(1.7, 1.068889) for Mjolnir, (1.7, 0.968889) for the other two - and:

```
reference   -38.239071  =  -39.939071 + 1.7    the tick-464 position
candidate   -38.270754  =  -39.970754 + 1.7    the tick-465 position
```

So the reference releases from the body position *before* this pass integrated it
and the port releases from the position after. Two candidate C# sites both read
`body.position` and both fire on a death: `SlimeController.cs:302`, which drops
the slot's `abilityPickupPrefab` as a `DynamicAbilityPickup` (kit-specific, which
fits the differing radii), and `PlayerCollision.cs:117-127`, whose
`AbilityExitInfo { position = body.position, isDead = true }` reaches the ability.
The kill itself is `DestroyIfOutsideSceneBounds` (cause 5,
`DestroyIfOutsideSceneBounds.cs:129-141`), and on `Player.prefab` that component
is `hierarchyNumber` 4 while `PlayerBody` is 5 - **the boundary component runs
before the body moves**, which is a mechanism for reading a pre-integration
position without any extra pass of history.

The site is pinned, and it is not the pickup drop. `flags 2307` is
`E_ACTIVE | E_ENABLED | E_PROJECTILE | E_EXPLOSIVE`, and the five abilities
`release_held_throw_item_on_exit` handles -
`ABILITY_GRENADE | ABILITY_SMOKE | ABILITY_TACK | ABILITY_MJOLNIR | ABILITY_POKE_BALL` -
are exactly the abilities held in the divergent kits. It is the `ThrowItem2`
payload, released because the holder died before `Fire`. The differing y offsets
resolve too: the held position is
`slot.base.position + local.complex_mul(basis)` where `local` is `gd.ground_position`
or `gd.air_position` from each ability's own throw config, so (1.7, 1.068889) for
Mjolnir and (1.7, 0.968889) for the other two is simply two different configs.
Nothing unexplained there after all.

**And this is where it gets interesting, because the port already does the thing
the obvious fix would do.** `throw_item_tick` calls `ability_boundary_kill` *before*
either of its two `slot.base.position +=` integrations, under a comment that says
so: "ThrowItem2's DestroyIfOutsideSceneBounds runs after the item logic and before
PlayerBody integrates this pass." The kill returns early, so nothing moves the body
afterwards, and `reported_position` is the pre-integration position of this pass.
That is already correct by the prefab ordering - and it is still one pass too late
to match.

Read against the exported rows: the reference releases from the position exported
at tick 464 and the port from the one exported at 465, and the exported player
position is itself published at the *start* of `throw_item_tick`. So the reference
is reading a position from before the *previous* pass integrated, not merely before
this one. That is precisely the residual fixture #8 could not explain, arrived at
independently, from a different ability, a different spawn site, and a code path
that already implements the pre-integration read correctly.

Two independent abilities needing "one pass further back than pre-integration" is
no longer a quirk of Black hole. It says this port integrates the body a pass ahead
of where the oracle does, and the local fixes - fixture #8's `spawn_origin` and
whatever lands here - are each paying for the same underlying offset. Worth
answering once. Do not patch a third spawn site without first testing whether the
pass boundary itself is misplaced.

`tack` carries one extra field the other two do not: `entity[1].owner`, reference
1 and port 0, on the released object. A concrete, reproducible instance of the
`entity[].owner` divergence that fixture #7 records as a general open defect.

### 2. Missile - the released object is published a step further on - 1 case

**Status:** FIXED 2026-09-24, see fixture #11. It was the same defect as
class 1: `Missile.prefab` is boundary/ability 5/6, and the port steered the
projectile before running the boundary. The guess recorded below - "the same
one-pass difference as class 1, seen in velocity and rotation instead of
position" - was right, including the caveat that it had not been traced.

`missile-early`, tick 371, which is its death tick. `entity[5]` (layer 10,
`kind 1`):

```
rotation   ref        245213 (0.000057)   cand   26807390223 (6.241525)
vx         ref       -3538962 (-0.000824) cand    3215404632 (0.748645)
```

The reference publishes the object essentially at rest and upright; the port
publishes it rotated -0.041660 rad (6.241525 is 2*pi - 0.041660, so it wrapped)
and moving at 0.748645. -0.041660 rad in one tick is -2.5 rad/s, an ordinary
angular velocity. So this reads as the same one-pass difference as class 1, seen
in velocity and rotation instead of position: the reference publishes the released
object before anything integrated it, the port after one step. Consistent with,
not proof of - I have not traced the missile's release site.

`late` and `brink` are clean, so it needs the `early` engage height.

### 3. Rewind's slime timer, exactly one tick - 2 cases

`rewind` on `early` and `late`: two `slime[0].timers[]` entries off by 71582788 =
**0.016667 = 1/60 exactly**. A timer started or decremented one pass apart.
Positions never diverge, so nothing downstream has consumed it yet. Nobody dies in
these two cases (`death reference 0 candidate 0`), and the divergence appears at
450 and 465 against an engage at 348, so this one is genuinely mid-flight and not
part of the death-tick group.

### 4. Spawn-hierarchy counter off by one or two - 5 cases

`gold_mushroom` on all three phases, diverging at **tick 0**:

```
entity[0].hierarchy  ref 100000023  cand 100000024
entity[1].hierarchy  ref 100000046  cand 100000048
entity[4].hierarchy  ref 100000016  cand 100000017
entity[5].hierarchy  ref 100000029  cand 100000030
entity[6].hierarchy  ref 100000039  cand 100000041
```

plus `platform` on `early` and `late`, one field each - and nobody dies in those
two, so they are not death-tick cases either. Nothing else differs,
positions included, so `next_spawn_hierarchy` advances by a different amount for
these prefabs. Do not file it as cosmetic: `Updater` orders a registration batch
by `hierarchyNumber` (`Updater.cs:271`), which makes this the input to every
ordering defect in this file - fixtures #6 and #8 both. On these cases it has not
yet changed an outcome.

### 5. A different number of objects released by the death - 13 cases

The `{3, 13, 16}` kit (Grenade, Smoke, Roll) in all three slot orders x 3 phases,
plus `beam-brink`, plus `grapple` x 3. All thirteen fire on the death tick.

I first read this as "the port spawns an extra entity", and that is wrong.
`beam-brink` has `header.entity_count` ref 10 / cand 9 while `smoke` and
`grenade` have ref 12 / cand 13 - the sign is not consistent. And the rows are
not shifted, they are **permuted**: `grenade-late` rows 3, 4, 5 carry
`source_ability` 0, 3, 13 in the reference and 13, 0, 3 in the port, the same
objects in a different order, several with `owner` swapped 2/0. Exported rows are
a reused pool, so row order is the pool's allocation history; a permutation plus a
count difference of one is that history diverging on the tick of the death. The
76 to 82 "differing fields" are one diverging allocation counted once per field of
every row it displaced, which is exactly the inflation `dr_report_all` was built
to make visible rather than to hide.

**This is probably not the `header.entity_count` defect fixture #7 records**, and
I should not have written that it was. That one fires at ticks 57-112 on ~5% of
maps and was measured loadout-independent (`DR_POOL=30` against `DR_POOL=2`,
identical 0/124 and 6/124). These thirteen fire on the death tick and only for
particular kits. Same field, different defect, until something measures them
together.

`grapple` is the same kit as `beam` (`{27, 28, 9}` against `{28, 27, 9}`) and
shows only a tail of it: 11 to 39 rope-node rows (`kind 3`, `layer 0`,
`owner 1`) where `flags` is 35 in the reference and 3 in the port. 35 - 3 = 32,
one bit, set on every node and on no other field, so it is a single per-node flag
the port never sets on the tick the rope's owner dies - worth fixing on its own
and independent of the permutation.

**Closed 2026-09-24 under fixture #10**, which took all thirteen and got eleven of
them to `FIRST any none`. It needed five fixes rather than the one or two this
description implies, and the guess that `grapple` and `beam` being one kit in two
slot orders meant something turned out to be a coincidence - measured, by landing
the rope fix alone and watching every count case stay character-for-character
identical. The four that remain, `grenade` and `smoke` on `early` and `late`,
reduce after the count fix to gaps byte-identical to class 1's, so class 1 and
this half of class 5 are **one** defect and class 1 covers ten cases, not six.

### 6. Last-bit noise, not defects - 3 cases

`meteor` early and late (`entity[4].x`, `.y`) and `meteor-brink`
(`player[0].x`, `.y`): gap of **exactly 1 raw unit**. Recorded so nobody spends a
day on them. Worth noting that `meteor-early` and `meteor-late` first differ at
348 and 363 while the death is at 469 and 484 - a last-bit residue that rides a
hundred ticks without growing is itself evidence the fixed-point arithmetic is
sound.

### Reproduce

```bash
BCT_ABILITIES=37,32,36 BCT_SLOT=0 BCT_AIM=0 BCT_ENGAGE_Y=-8.0 BCT_ALL=1 \
  native/artifacts/rl/bounds_case_trace \
  "$PWD/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so" \
  "$PWD/native/target-q32/release/libbopl_sim.so"
```

The ability triples and slot/aim columns are the case table in
`eval/rust_ab/bounds_ability_ab.c`; the three phases are `BCT_ENGAGE_Y` 0.0,
-8.0 and -10.9. `BCT_DUMP_ALL=1` additionally dumps every exported row of both
sides, not only the owned ones, which is how the fixture #8 entity was
identified.

This sweep is deliberately **not** a gate. Making it one means deciding which of
these six classes is allowed to stay, and four of them should be fixed instead.

## Fixture #9 - ThrowItem2's boundary component runs before the ability's own pass, not after

**Status:** FIXED 2026-09-24 on branch `throwitem-pass`. Closes class 1 of the
114-case trajectory sweep - all six cases now report `FIRST any none / FIRST
discrete none` for the whole episode - and, more usefully, **retires the
"misplaced pass boundary" hypothesis that class 1 and fixture #8 together
seemed to support**. See "What this says about fixture #8" below.

Before, rebuilt from committed source at `847d7c9b`:

```
mjolnir-early  465 NUMERIC  entity[0].x ref -164235558265 (-38.239071) cand -164371638025 (-38.270754)
                            entity[0].y gap 545057382 (0.126906)
mjolnir-late   393 NUMERIC  entity[0].x ref -172798692450 (-40.232831) cand -172934698609 (-40.264497)
poke_ball-early 465 NUMERIC entity[1].x  same gaps
poke_ball-late  393 NUMERIC entity[1].x  same gaps
tack-early     465 DISCRETE entity[1].owner ref 1 cand 0, plus the same x/y gaps
tack-late      393 DISCRETE entity[1].owner ref 1 cand 0, plus the same x/y gaps
```

After: `FIRST any none / FIRST discrete none` on all six, at the reference's own
death tick. Two defects, not one, and they are independent - the position fix
alone left tack's `owner` as the sole remaining field on both tack phases, which
is how they are attributed here.

### 1. The ordering was simply backwards on the prefab

The port called `ability_boundary_kill` at the *end* of `throw_item_tick`, under
a comment asserting "ThrowItem2's DestroyIfOutsideSceneBounds runs after the
item logic and before PlayerBody integrates this pass". The first half of that
is false. All five ThrowItem2 prefabs - `Mjolnir.prefab`, `Poke ball.prefab`,
`Banana peel.prefab` (Tack), `Smoke.prefab`, `Grenade_0.prefab` - carry the
byte-identical component order:

```
1 SpriteAnimator  2 Ability  3 PlayerCollision  4 PlayerPhysics
5 DestroyIfOutsideSceneBounds  6 ThrowItem2  7 PlayerBody  8 DPhysicsBox
```

DestroyIfOutsideSceneBounds is **5** and ThrowItem2 is **6**, so the boundary
component runs ahead of the *entire* ability component, not merely ahead of the
integration. On the pass it fires, `ThrowItem2.UpdateSim` never runs at all.

That matters because `ThrowItem2.OldUpdate` - the first thing `UpdateSim` calls -
republishes the held payload every pass:

```csharp
else if (!hasFired && !grenade.hasBeenThrown)
{
    grenadeBody.position = dummyPos();          // ThrowItem2.cs:282-285
}
...
private Vec2 dummyPos()                          // ThrowItem2.cs:288-293
{
    ...
    return Vec2.ComplexMul(new Vec2(body.rotation), u) + body.position;
}
```

So the held payload's position is a per-pass write from `body.position`, and
`body.position` is a live property over `fixTransform.position`
(PlayerBody.cs:42-51), not a cached field. On the death pass that write is
skipped, and the payload is released carrying the position written on the
*previous* pass. `DestroyIfOutsideSceneBounds.selfDestruct` resolves
`IAbilityComponent` and calls `ExitAbility` directly
(DestroyIfOutsideSceneBounds.cs:129-141), reaching `ThrowItem2.ExitAbility`,
which calls `ManualInit` and `AddForce` on the payload and never touches its
position (ThrowItem2.cs:179-193).

That closes the arithmetic the sweep recorded. At `early`, player 0 exports
(-39.939071, -11.182124) on tick 464 and (-39.970754, -11.309030) on 465, the
tick it dies; Mjolnir's release offset is (1.7, 1.068889):

```
reference   -38.239071  =  -39.939071 + 1.7    the tick-464 exported position
candidate   -38.270754  =  -39.970754 + 1.7    the tick-465 exported position
```

The exported player position for tick N is published by `Ability` at
hierarchyNumber 2, before PlayerBody at 7, so it *is* the pre-integration
position of pass N - which is exactly the value `dummyPos()` reads at
hierarchyNumber 6 of that same pass. Reference = the payload's pass-464 write.
Port = its own pass-465 write, because the port ran its boundary check last.

**The fix** (`native/src/rollout/mod.rs`) moves the single
`ability_boundary_kill(index, false)` call from the end of `throw_item_tick` to
the top, immediately after the `Ability`-ordered `reported_position` publish and
ahead of every piece of ThrowItem2 logic. No new state, no history field, no
shim. The kill *tick* and kill *position* are unchanged by construction: nothing
between the top of the function and the old call site writes `base.position` on
any path that reached the old call site, which is why `bounds-ability-ab` reports
the identical 38 cases at the identical ticks and positions.

Everything else ThrowItem2 does on the death pass is now skipped as well - the
grounded-exit test, the aim/facing update, the fire-button edge, the tick and
`timePassed` counters, the held-payload spawn and the phase-1 exit. Only the
position was measurable in these six cases; the rest follow from the same
ordering and were wrong for the same reason.

### 2. Tack's released row lost the owner it had already earned

`tack` carried one extra differing field the other two kits did not:
`entity[].owner`, reference 1 and port 0. It is a real second defect and it does
not share a cause with the position, but it is the same *event* - what the
release publishes - so it is fixed here.

The exported `owner` is `transform.GetComponent<IPlayerIdHolder>().GetPlayerId()`
(`sim/Gameplay/RlObservation.cs:110,118`). For the Tack payload that id is bound
in BananaPeel's first updater pass, which the port models with `ThrownItem::
launched`. `tack_items_tick` takes that edge for a *held* payload - unlike
`mjolnir_items_tick` and the Poke ball pass, both of which skip an item whose
collider does not exist yet - so by release time `launched` is already true and
the `self.entities[entity].owner = item.owner + 1` write inside the edge has
been spent while `entity` was still `None`. `release_held_throw_item_on_exit`
then built the row from `CEntity::default()`, i.e. owner 0, and nothing ever set
it again.

Letting Tack launch while held is correct and must not be "fixed": the same edge
sets `angular_velocity = 4`, and the released row carries `rot 0.066667` = 4/60
in **both** backends, so the reference's held banana peel does go through that
pass. What was missing is only the publication. The fix gives
`release_held_throw_item_on_exit` the owner expression `fire_thrown_item`
already uses verbatim - `if ability == ABILITY_TACK && item.launched { base_id }
else { 0 }` - under the rule its own comment states: a payload held through
BananaPeel's first pass is already bound when `ManualInit` registers the
collider.

Mjolnir and Poke ball need nothing here, and the measurement says so: both
report `owner 1/1` even before this change. Their item passes gate the `launched`
edge on the collider existing, so the edge is still unspent at release and fires
later in the same tick, after `release_held_throw_item_on_exit` has allocated
the row.

### What this says about fixture #8, which was the point of the exercise

**It dissolves the shared-cause argument, and fixture #8's own question stays
open.** The sweep's class-1 note concluded that because the port already
implemented a pre-integration read here and was *still* one pass late, "this
port integrates the body a pass ahead of where the oracle does", and that
fixture #8's `spawn_origin` and whatever landed here were paying for one
underlying offset. That is not what happened. The port's pre-integration read
was correct and irrelevant: the position in question is not read at the kill
site at all, it is written once per pass by a component that the boundary
pre-empts. An ordinary in-pass ordering error, fixed by moving one call, with no
extra pass of history anywhere.

So the "two independent abilities need one pass further back" coincidence is
down to one. Black hole is again alone, and the pass boundary is **not**
implicated by this evidence. Do not carry class 1 forward as corroboration.

One hypothesis for fixture #8 is also closed on the way past, because it is the
obvious next guess: `PlayerBody.position` is **not** a cached field that would
lag its transform. It is a property that reads and writes
`fixTransform.position` directly (PlayerBody.cs:42-51), so reading `body.position`
from an animator callback at hierarchyNumber 1 gives the live value, and "the
spawn read a stale PlayerBody copy" cannot be the explanation for Black hole.

### Negative results worth keeping

- **Lead 1 as originally framed was half right.** The brief asked to find "what
  else publishes the held row each pass", on the theory that `spawn_held_item`'s
  one-time `position:` write could not explain a moving row. The per-pass write
  does exist and is the right object - `throw_item_tick`'s
  `self.thrown_items[item_index].position = slot.base.position + dummy.complex_mul(...)`
  block - but it was never running "after the body integrates". It runs before,
  correctly. The defect was that on the death pass it ran *at all*.
- **Lead 2, the misplaced pass boundary, was not needed and is not supported.**
  No broad change was attempted and none is warranted by these six cases.
- **The offsets are not part of the bug**, as the brief said. (1.7, 1.068889) for
  Mjolnir against (1.7, 0.968889) for the other two is two throw configs, and the
  identical raw gaps across the three kits are just "one tick of the same
  player's motion" at a shared death tick.

### The ordering is worth checking on the other forms

Resolving `hierarchyNumber` against the script guids on every ability prefab
gives the boundary component's position relative to the ability's own script.
The port's call sites carry comments in both directions, and the gun family's
comment - "GunTransform's DestroyIfOutsideSceneBounds runs after the fire test"
- is **correct**: on `Blink gun`, `Duplicator`, `Freeze ray`, `Growth ray`,
`Shrink ray` and `Magnet gun` the gun script is hierarchyNumber 2 and the
boundary is 6. The ThrowItem2 comment looks like that one copied to a prefab
where it does not hold.

Read the pairs below as `boundary/ability`, always in that order, never as
"smaller first" - the two lists run in opposite directions and a bare inequality
is exactly the ambiguity that produced this bug. Spot-checked against the prefab
YAML by resolving guid `6e66898249f8b2c751f742b36806036a`: Gust boundary 4,
Roll boundary 5, Meteor boundary 5, all three as listed.

Forms where the boundary runs **before** the ability script, i.e. where the
port's kill must come first: Mjolnir/Poke ball/Banana peel/Smoke/Grenade_0
(5/6, fixed here), Black hole (5/6), Bow (5/6), Mine (5/6), Missile (5/6),
Gust (4/6), Engine (4/6), Push (4/6), Spike (4/6), Revival (4/6),
Tesla coil (4/6), Throw (5/8), Beam (4/8), Drill (3/7).

Forms where it runs **after**, i.e. where the ability's own update must run
first: the six guns above (6/2), plus Meteor (5/3), Platform (5/2), Roll (5/4),
Time stop (5/1), Dash (5/1).

**Sweep class 2, `missile-early`, was this same defect and is now fixed** in
fixture #11, together with the Mine comment named below. Both are recorded
there rather than here.

### The whole 114 re-run against the base, which is the regression evidence

The eight gates are green, but none of them reads a mid-flight trajectory, so
they cannot by themselves say this change moved only what it was aimed at. So
the 114-case sweep was run twice on the same machine and the same oracle: once
on `847d7c9b` rebuilt from source, once on this change, with identical
parameters.

```
base    clean=70  divergent=44  total=114
fixed   clean=76  divergent=38  total=114
```

The divergent **sets** differ by exactly the six cases this fixture is about -
`mjolnir`, `poke_ball` and `tack` at `early` and `late` - and by nothing else.
Every one of the 38 surviving divergences reports a **byte-identical**
`FIRST any` line before and after, so no other case moved even in its last bit.

One caveat on the numbers, because they do not match the 84/114 recorded in the
sweep section above: this pair was run at `BCT_AIM=0`, which is what the sweep's
own "Reproduce" block and the class-1 measurements use, while the prose there
says the aim column comes from the case table in `bounds_ability_ab.c`, where it
is 64. The two parameterizations give different totals - the aimed-gun family of
fixture #1 is aim-sensitive, and at `BCT_AIM=0` `plain`, `duplicator`, `blink`,
`freeze`, `growth_ray`, `shrink_ray` and `magnet` pick up entity-basis
divergences that the aim-64 run does not show. **Do not compare 76/114 against
84/114.** The base-versus-fixed pair above is internally consistent and is the
claim being made.

**Settled afterwards, in fixture #11: the recorded sweep was `BCT_AIM=64`.** The
114 re-run at aim 64 measures 24 divergent on this change. `847d7c9b` itself was
not re-measured at aim 64, so the corroboration is arithmetic and not direct:
the record's own 30, minus exactly these six, is 24. So the sweep section's
"Reproduce" block is the wrong half of that contradiction, and its totals should
be read as aim 64. Class 1 reproduces at either aim, which is why nothing looked
wrong when it was measured at 0; class 2 only reproduces at 64.

### Reproduce

```bash
BCT_ABILITIES=37,32,36 BCT_SLOT=0 BCT_AIM=0 BCT_ENGAGE_Y=0.0 BCT_ALL=1 \
  native/artifacts/rl/bounds_case_trace \
  "$PWD/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so" \
  "$PWD/native/target-q32/release/libbopl_sim.so"
```

The six cases are `BCT_ABILITIES` 37,32,36 (Mjolnir), 33,2,14 (Poke ball) and
35,2,14 (Tack), each at `BCT_ENGAGE_Y` 0.0 and -8.0.

## Fixture #11 - Missile steered its projectile one pass past the death, and Mine had the same reversed comment

**Status:** FIXED 2026-09-24 on branch `throwitem-pass`, on top of fixture #9.
Closes sweep class 2. Two sites, one of which no case measures.

This is fixture #9's defect class a second and third time, which is the point of
the entry: the ordering claim in a comment is what goes wrong, and it goes wrong
by being copied from a prefab where it held to one where it does not.

### Missile - the measured one

`Missile.prefab` is `boundary/ability` **5/6** - DestroyIfOutsideSceneBounds 5,
SparkShot 6, DPhysicsBox 7, PlayerBody 8 - so the boundary runs ahead of the
whole of SparkShot's update. `missile_form_tick` ran `steer_active_missile`, the
phase machine and `spawn_guided_missile` *before* `ability_boundary_kill`, so on
the death pass the port advanced the live projectile one step the oracle never
takes.

Before, `missile-early`, nine differing fields on `entity[5]` (layer 10, kind 1)
at tick 371, which is the death tick both backends agree on:

```
entity[5].rotation  ref        245213 (0.000057)  cand  26807390223 (6.241525)
entity[5].vx        ref      -3538962 (-0.000824) cand   3215404632 (0.748645)
entity[5].x         ref -158228846144            cand -158175197084  gap 53649060
...plus y, vy, right_x, right_y, up_x, up_y
```

6.241525 is 2*pi - 0.041660, i.e. -0.041660 rad wrapped - one tick at -2.5 rad/s,
an ordinary angular velocity. After: `FIRST any none` on all three phases, at the
reference's own death tick.

Read that field list as a **lower bound, not an inventory**: `BCT_ALL=1` prints
every differing field of the *first* diverging tick and then stops, so it says
what went wrong first, not everything that went wrong. The clean/divergent split
is trustworthy in both directions - `FIRST any none` means no tick of the episode
differed in any compared field - but a nine-field list does not mean the episode
contained nine bad fields.

The fix moves the single `ability_boundary_kill` call to the top of
`missile_form_tick`. It cannot move the death tick: `steer_active_missile` writes
only `guided_missiles[..]` and `spawn_guided_missile` only reads the body, so
`base.position` is identical at the old and new call sites. The
`Ability`-ordered `Player.Position` publish happens in the shared player loop
before the ability dispatch, so the kill takes the right position directly.

**This case needs `BCT_AIM=64` and is invisible at `BCT_AIM=0`**, because
`steer_active_missile` reads `action.direction()` and a zero aim leaves
`next_up = missile.up`, so nothing is steered and there is nothing to get wrong.
That matters beyond this fixture - see the aim note below.

### Mine - the unmeasured one, and the comment that was the actual bug

`mod.rs` carried, at the `mine_throw_tick` kill site:

> ThrowItem's DestroyIfOutsideSceneBounds runs after the mine logic and before
> PlayerBody integrates this pass.

`Mine.prefab` is `boundary/ability` **5/6** - DestroyIfOutsideSceneBounds 5,
ThrowItem 6, DPhysicsBox 7, PlayerBody 8 - so the boundary runs **before**
ThrowItem's update and the first half of that sentence was reversed, exactly as
ThrowItem2's was.

**The code needed to change too, not only the comment.** The kill ran after the
entire mine update, `fire_mine` included, so on a death pass the port could
spawn a mine the oracle never spawns, or take the phase-2 exit instead of dying.
That is the same observable class as fixture #9's released payload. It is latent
only because no case in the 114 reaches it: `mine` is clean on all three phases
at both aims, before and after, and its death ticks (371/367/367) are unchanged.
So this half of the change is **unmeasured by construction** - it is made because
the prefab says so and because leaving known-wrong ordering behind a corrected
comment is worse than either alone, not because a number moved.

### The 114 re-run, and what it finally settles about the aim column

Run against the pre-change build (`46d513f7`) exactly as fixture #9's was:

```
aim=0    pre 38 divergent   post 38 divergent   set difference: none
aim=64   pre 24 divergent   post 23 divergent   set difference: exactly missile-early
```

No regressions at either aim, and every surviving divergence reports a
byte-identical `FIRST any` line before and after.

**The aim ambiguity flagged in fixture #9 is now resolved: the recorded sweep was
`BCT_AIM=64`.** The arithmetic closes on the record's own number - 30 divergent
at `847d7c9b`, minus the 6 fixture #9 closed, is 24, which is exactly the
pre-change count measured here; minus `missile-early` is 23. The sweep section's
"Reproduce" block, which says `BCT_SLOT=0 BCT_AIM=0`, is the wrong half of that
contradiction and should be read as 64 for anything reproducing the sweep totals.
Class 1 happens to reproduce at either aim, which is why fixture #9's six cases
were measured at 0 without anything looking wrong; class 2 does not, and that is
the case that exposes it.

Anyone re-running the sweep should use `BCT_AIM=64` and expect 23. At
`BCT_AIM=0` the totals differ in both directions - the aimed-gun family of
fixture #1 picks up entity-basis divergences that aim 64 does not show, and
`missile` loses its only divergent phase - so the two are not comparable and
neither is wrong, they are different cases.

### Reproduce

```bash
BCT_ABILITIES=10,2,14 BCT_SLOT=0 BCT_AIM=64 BCT_ENGAGE_Y=0.0 BCT_ALL=1 \
  native/artifacts/rl/bounds_case_trace \
  "$PWD/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so" \
  "$PWD/native/target-q32/release/libbopl_sim.so"
```

`late` and `brink` are clean, so this needs the `early` engage height as well as
the non-zero aim.

## Open items inherited from the deleted porting docs

Not found by this sweep, but this is where open defects live now. The 2026-09-23
reorganization deleted `sim/PORTING.md`, which carried these as unticked
checklist entries; they are recoverable in full from git history.

**The oracle throws on one recorded game** - the map-27 `ShootDuplicator`
NullReferenceException. Documented under fixture #4, where it was long
misattributed to the port.

**Determinism flake, 1 in ~17 runs, never reproduced.** One `checkpoint.sh` l4
replay diverged at tick 0 on `playerDied` + `updatables` and 16 retries failed to
reproduce it. The suspicion was a `Dictionary<string,_>` iteration-order leak
(.NET randomizes string hash seeds per process) on a trace-observable path; a
later guess was a socket-path artefact. Both rest on that single observation, so
the rate on the in-process path the RL evaluation actually uses is unknown. This
matters for any oracle-hosted measurement: if the backend is not reproducible
across processes then per-game comparisons are noise and only large-N averages
mean anything. `eval/backend_determinism.py` is the instrument that measures
it - each repeat is a separate `evaluate.py` process, which is the unit the
hypothesis is about - and it should be run on the Rust sim too, where the arm
must come back perfectly identical.

**AOT residual risk in the coordinator.** Serialized `List<T>` fields without
inline initializers, on maps and loadouts not yet exercised, may need `rd.xml`
entries. A serialized-field scan that generates `rd.xml` would close it.

**Six latent trajectory divergences in `bounds-ability-ab` - all now closed.**
The three Gust lines are fixture #5 (fixed 2026-09-24); the three Black hole
ones are fixture #6 (fixed 2026-09-24). **Five of the six were invisible until
2026-09-24.** The gate had always reported them - the case table marked Black
hole and Gust `known_trajectory_divergence` and printed them without failing,
because the probe's subject is *when the boundary kills*, not the trajectory
into it. What hid them was a stale binary: `native/artifacts/rl/` is gitignored,
`make` compares mtimes, and the compiled gate was older than its committed
source, so it printed 24 ability lines and 1 divergence where the rebuilt gate
prints 114 and 6. The summary line is identical either way (`cases=38
water=-11.3 survivors=4`), so nothing looked wrong. **Delete the gate binaries
before trusting a gate run** - the binaries, not the directory. `rm -rf
native/artifacts/rl/` also destroys the 5.9 GB uv-built training venv that lives
there, which nothing in the Makefile provisions and which `check-routing` and the
`replay_ab` selfcheck both need (`Makefile:193`, `:324-326`); that cost an agent a
failed gate on 2026-09-24. `train/venv-snapshot.txt` records what is in it. Use
something like:

```bash
find native/artifacts/rl -maxdepth 1 -type f -executable -delete
```

Rebuilt on `c2293d22`:

```
early blackhole  reference 404 (-36.1894,-11.3380) | rust 404 (-36.7478,-11.3380)
early gust       reference 400 (-36.1016,-11.7544) | rust 400 (-36.1999,-11.7727)
late  blackhole  reference 391 (-41.0446,-11.3957) | rust 391 (-41.4578,-11.3957)
late  gust       reference 395 (-40.9649,-11.5982) | rust 394 (-41.0592,-11.3426)
brink blackhole  reference 367 (-42.2811,-11.6208) | rust 368 (-42.2970,-11.6558)
brink gust       reference 367 (-42.2811,-11.6208) | rust 385 (-42.3677,-11.6709)
```

The source comment's reading of Black hole - y bit-identical in all three
phases, kill tick matching in two, x drifting 0.4-0.56 - was accurate but
described a symptom; the cause was the missing per-pass `AirealMovement` damping
on the cast form, plus a swapped publication order that the kill had been
hiding. **Gust's comment was wrong**: it said the port "crosses the plane one
tick early", which describes `late` only; on `brink` the port survived **18
ticks longer** than the reference (367 vs 385), fifty times the documented gap
and in the opposite direction. Both forms turned out to carry two defects each,
sharing the entry-pass boundary component. No case carries
`known_trajectory_divergence` now, so a regression in any of the 38 fails the
gate rather than printing. One of the two residues left behind in the Black hole
path turned out to be a third defect, not rounding - fixture #8. The other, a
~1e-5 rad rotation error on a grounded cast, is still open under fixture #6's
negative results.

**The Unity player build is a week older than the observation ABI it must
speak.** `Builds/LinuxIL2CPP/BoplBattle.x86_64` was built 2026-09-07. The
handshake in `sim/abi/bopl_rl.c:488` requires protocol version 3, which landed
2026-09-14 in `3721f8ae` ("rl: wire observation ABI v3"), so the player is
rejected. `Tools/play-vs-agent.sh` and `Tools/play-vs-agent-mac.sh` are dead
until it is rebuilt, which needs a machine with Unity.

Two claims that used to stand here were measured on 2026-09-24 and are false.
Both are recorded because each one cost time.

*The rejected handshake does not dump core.* It reports `Unsupported RL
protocol: Protocol error` and exits 0. Verified with `ulimit -c unlimited` and
systemd-coredump active, 223 recorded dumps before and after - the first attempt
at this test ran under `ulimit -c 0`, which would also have suppressed a dump
from the spawned player, so it proved nothing and had to be redone.

*`check_routing.py` never needed the Unity player.* It takes `--game` and runs
against either backend library, so the stale player was not what stopped it; it
was broken on its own terms, and is now fixed (`4dd61e77`, `66fc2a16`). It
passes 31/31 against both the oracle and the port, so per-player routing through
PufferLib's agent permutation, the frozen bank slices, and the tag /
`count_aligned` alignment protocol do have an executing test after all. The
three bugs it turned out to carry are worth knowing about, because all three
were in the check rather than the thing checked, and the tell was that the
oracle and the port failed the *identical* eight checks - two independent
implementations do not share a bug. They were: reshaping observation rows at
`OBS_SIZE` when the row pitch is `BOPL_OBS_STRIDE`, which reads row 0 correctly
and every later row at a sliding offset; calling only PufferLib's reset and not
the `bopl_vec_reset` that follows it in the trainer, so it inspected a state the
trainer never trains from; and counting completed matches as `matches-1` when
the native reset makes the pre-stepping baseline 2.

Nothing gated `check_routing.py`, which is how it drifted through an
observation-layout change unnoticed. `make check-routing` now does, running
**both** backends, because the oracle and the port agreeing is the diagnostic
that separates a broken check from a real divergence.

It was proven to fail before it was trusted. Reverting each of the three fixes
individually, the gate fails with a distinct signature each time, and the
unmodified tree passes:

| reverted | gate | failing checks |
|---|---|---|
| row pitch | FAIL | 6 - the observation-content cluster |
| native reset | FAIL | 8 - the terminal flags *and* the routing checks, because without it the games are never restarted under the installed permutation |
| completion arithmetic | FAIL | 1, reporting `matches=40 completed=44` |
| nothing (HEAD) | PASS | 62 across both backends |

The clean arm earned its keep immediately: on the first attempt all four arms
"failed", including that one, which is what exposed the gate itself being wrong
rather than the code. `--output` becomes the native log path and `bopl_rl_open`
rejects a relative executable/replay/log, so the target needs
`$(abspath $(OUT))`. A three-arm proof without a clean control would have read
as a total success.

Worth noting alongside the handshake result above: *that* path aborts. A relative
path fails the startup check with `Absolute executable/replay/log paths and
timeout 1..3600000 ms required` and dumps core. So "does the ABI dump core" has
different answers for different rejections - the protocol-version rejection exits
0 cleanly, the path-validation rejection aborts.
