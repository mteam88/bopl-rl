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

`research/rust_ab/replay_ab.c` reuses one handle per backend across every
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

`make -C native/rl ability-sweep`
(`research/divergence/run_ability_sweep.sh`). Loadouts drawn uniformly from all
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
