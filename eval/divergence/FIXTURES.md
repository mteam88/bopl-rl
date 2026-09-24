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
kill, and they are not what this fixture fixed:

- A *grounded* cast diverges on the entry tick at `entity[2].right_x`, the
  hurtbox orientation, by 329 raw units (~7.7e-8). Reproduce with
  `BCT_ABILITIES=18,6,29 BCT_ENGAGE_Y=100`. This was confirmed pre-existing by
  building the unmodified base and running the same case: both builds emit the
  byte-identical `cand 5411486426` against `ref 5411486755`.
- On `early` and `late`, a *distant idle* player 1 diverges by ~71 raw units
  (~1.6e-8) on the tick the hole spawns - 16 passes after the cast, `early` 364
  and `late` 379. Attribution is rigorous even though the base build cannot show
  it directly (its own entry-tick divergence masks it): `dr_compare` walks
  header, then players, then slimes, so `header.entity_count`, `player[0].x/y`
  and `player[0].scale` all matching at that tick proves the hole spawned on the
  same pass from a bit-identical position and scale. The difference is therefore
  in the attraction the hole applies to other bodies - `black_holes_tick`, which
  this change never touches.

**The brief's framing was right about the count but not the content.** It
predicted two defects and named the second only as "a 0.4-0.56 unit horizontal
drift, undiagnosed". It also suggested checking whether Black hole's entry path
repeats Gust's stale-`attachedGround` `UnGround` bug. It does not:
`BlackHoleClap.OnEnterAbility` never calls `UnGround` at all (BlackHoleClap.cs:52-76),
and the port's entry arm already clears `ground` without a separation nudge. That
line of inquiry is closed.


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
before trusting a gate run.** Rebuilt on `c2293d22`:

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
gate rather than printing. Two small pre-existing residues in the Black hole
path remain open and are recorded under fixture #6's negative results.

**The Unity player build is a week older than the observation ABI it must
speak.** `Builds/LinuxIL2CPP/BoplBattle.x86_64` was built 2026-09-07. The
handshake in `sim/abi/bopl_rl.c:488` requires protocol version 3, which landed
2026-09-14 in `3721f8ae` ("rl: wire observation ABI v3"), so the player is
rejected with `Unsupported RL protocol` and the process dumps core rather than
exiting cleanly. Everything that drives the real game is therefore dead until
the player is rebuilt: `train/check_routing.py`, `Tools/play-vs-agent.sh` and
`Tools/play-vs-agent-mac.sh`.

That matters beyond convenience. `check_routing.py` is the only thing that
verifies per-player routing through PufferLib's agent permutation, the frozen
bank slices, and the tag / `count_aligned` alignment protocol that decides which
matches count towards per-bank pool statistics. While it cannot run, that whole
mechanism has no executing test, so a change to opponent scheduling is verified
offline or not at all. Rebuilding the player is the fix; the core dump on a
rejected handshake is a second, smaller bug worth closing at the same time.
