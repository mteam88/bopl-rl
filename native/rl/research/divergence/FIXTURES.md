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

## Fixture #2 - the port carries entity rows across `bopl_rl_reset`

**Status:** reproducible; affects every batched differential comparison in the
repo, including `replay_ab` and the map sweeps.

When one `BoplRl` handle plays several episodes back to back, the Rust port's
exported entity table can gain rows belonging to an *earlier* episode. The
oracle never does this.

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

## Fixture #3 - `replay_ab` results depend on how many replays it is given

**Status:** reproducible; root cause is fixture #2.

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

**Consequence.** `make replay-sweep` and the map sweeps are order-sensitive:
adding or removing a `.rep`, or changing glob order, can change the reported
first-divergence tick for unrelated replays. Anyone using those gates as a
regression signal should either reopen handles per replay or pin the file set.

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
