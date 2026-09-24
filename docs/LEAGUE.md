# The league

A permanent, committed record of which agents exist and how they do against
each other. Training runs join it, contribute members to it, and are measured
against it. It is not per-run scratch state and it does not die with a run.

Two files are the artifact. Everything else in this directory reads or writes
them.

## `league.json`

```
{ "schema": 1, "anchor": "random",
  "members": [ { "id", "name", "kind", "parent", "origin", "sim",
                 "reward", "loadouts", "added", "path?", "metadata_sha256?" } ],
  "conditions": { "<key>": { "key", "sim_sha256", "maps", "episode_ticks",
                             "delay", "loadouts", "loadout_sha256" } },
  "league_sha256": "..." }
```

- `id` — sha256 of the weights for a checkpoint, the bot name for a scripted
  bot, `random` for the anchor.
- `kind` — `main`, `exploiter`, `specialist`, `scripted`, `random`, `human`.
  A `human` member is one of Matthew's recorded games; it has no weights and
  cannot be scheduled by `play`, which refuses it with a message. Its counts are
  appended to `payoff.jsonl` from the recording.
- `parent` — the member this was forked or reset from; `null` for random init.
  Every exploiter here is `null` on purpose: exploiters start from random
  weights, not from a fork of the main, so they are free to find combinations
  the main's representation cannot reach.
- `target` — the member an exploiter was trained to beat. This is a *different*
  question from `parent` and does not substitute for it: because our exploiters
  begin at random init, `parent` is null by design, so anything asking "how long
  did this exploiter take to beat what it was aimed at" must read `target`.
  Only `exploiter` and `main` may carry one; any other kind is rejected.
- `origin` — `{run, step}` for trained members.
- `sim` — the simulator library the member *trained* under. This is provenance,
  and is not the same thing as the library its games were *played* under, which
  lives in the condition.
- `reward` — `{tag, binding_sha256, shaping}`. Reward semantics live in the
  training binding, not the simulator, so `native_binding_sha256` is what
  actually discriminates them.
- `added` — `{date, rule}`: when, and which rule admitted it.

Members are immutable. `league_sha256` covers the member list, and every
checkpoint's weights are re-hashed on load; either mismatch refuses the load.

## `payoff.jsonl`

One JSON object per line, append-only:

```
{"a": "<member name>", "b": "<member name>", "condition": "<key>",
 "wins": <a's wins>, "losses": <a's losses>, "draws": <draws>}
```

Aggregated counts only — never per-game rows. Bradley-Terry and Nash need only
counts, and a league that grows for months would otherwise commit millions of
lines to git permanently. Per-game detail is derived data and belongs under
`native/artifacts/`.

`a` and `b` are member *names*, unique and permanent.

**Ordering is canonical: `a < b`, always, with the counts taken from `a`'s
side.** A matchup therefore appears under exactly one ordering; `(b, a)` rows do
not exist. A consumer may sum every row with a matching `(a, b, condition)`
without inspecting direction and will not double-count. `registry.canonical` is
the single place this is enforced, at the only point rows are written.

Do not confuse this with side-swapping. Every matchup is played from *both*
seatings, and both seatings are already folded into these counts.

`wins`, `losses` and `draws` are each written explicitly from the games' own
outcomes. Never reconstruct a loss as `games - wins`: that silently converts
every trade and every timeout into a loss for one side, which is the exact bug
that made a kit's "92.6% win rate" really 62.2% wins plus 30.4% trades.

Counts are **never summed across conditions**.

## What a condition is

`sim<sha8>.<loadouts>.<maps>.t<ticks>.d<delay>` — the simulator library, the
loadout distribution, the map set, the episode limit and the observation delay.
That is everything that can change who wins a game.

The map part is a label, not the data: the full list is recorded in the
condition. A set that is exactly one of `native/data/map-sets.json`'s named sets
is named (`maptraining-111`), up to eight maps are spelled out with separators
(`map0-1-2`), and anything larger is an 8-hex digest of the list (`map8168a8f1`).
The separators are not decoration - the first form concatenated the IDs, so
`[1, 12]` and `[11, 2]` both read `map112` and two different conditions would
have pooled their counts, which is the one thing a condition exists to prevent.
The two conditions measured before this was fixed were renamed from `map012` to
`map0-1-2` in league.json and in all 189 payoff rows; no count moved.

The reward function is deliberately **not** part of it, though it is recorded on
every member. Reward decides what a policy was trained to maximise, not who wins
a game that is played: outcomes are read off the simulator's own end-of-episode
flags. `f65c33b7` changed only the training binding and says of the simulator
"Nothing in native/src/rollout or sim/ moves", so games either side of it under
one library are directly comparable. Keying them apart would split the matrix
into halves that could never be pooled and cost twice the CPU to fill.

## Scoring

Draw-aware, always: a win is 1, a draw 0.5, a loss 0, computed from each game's
final player flags via `eval/elo_ladder.play_games`. The legacy `win` field
counted a mutual death as a win for whichever side wrote the report and is never
read. Games are always side-swapped pairs on fixed seeds.

## Commands

```
python league seed     # write league.json: specialists, scripted roster, anchor
python league play     # play matchups, append counts   (the only verb that costs CPU)
python league rate     # Bradley-Terry Elo, Nash averaging, RPP
python league status   # members, matrix coverage, saturation
python league add      # append one immutable member
python league check    # prove the estimators on matrices with known answers
```

`play` gates matchups it has already decided: a pair whose whole confidence
interval sits outside 15–85% is skipped as uninformative, which is where the
throughput goes. Re-running `play` deepens only what is still in doubt.

There is exactly one rating scale in the repo: `eval/elo_ladder.py`'s
`bradley_terry_fit` (draw-aware, `random` anchored at 0 Elo, with intervals).
This package calls it and does not reimplement it.

## Training against the league

A run's PFSP opponents are its own frozen snapshots *and* league members, drawn
from one distribution on one statistic. `--opponents` chooses the members:

```
--opponents league              # every member with weights (the default)
--opponents self                # snapshots only, the old behaviour
--opponents drill-throw-magnet  # one member: this is what an exploiter is
```

There is no share to set. A league member is a permanent PFSP candidate and a
snapshot is an evictable one (`--pool-size` bounds the snapshots, not the
candidate set), and PFSP weights all of them by the learner's own draw-aware
score against each. An exploiter is not a separate mechanism, a separate loop or
a separate flag family: it is a run from random init whose `--opponents` names
one member.

Members are identified by name everywhere the run reports them - `pool_opponents`
in `metrics.jsonl`, `pool.banks[].opponent`, and the `intended`/`realized`/`played`
shares in `pool.pfsp` are all keyed by name, with a run's own snapshots named
`self@<physical decisions>`.

**A member can only be loaded by a run shaped like it.** A frozen bank is a flat
float32 array that the learner's own kernel indexes, so nothing about loading one
is type-checked: a 4.4 MB v4/256 checkpoint read into a 620 KB set/128 policy is
accepted, and the opponent plays at chance. That is not hypothetical - the first
run of this feature reported an even score against `drill-throw-magnet`, a member
rated 592 Elo above random, and the scores were even against all eight members
because all eight were noise. `registry.opponents` now compares the member's
checkpoint sidecar against the run's own architecture (`architecture`, `obs_size`,
`hidden`, `num_layers`, `aim_fourier`, `aim_ar`, `v4_shape`) and refuses a
mismatch before the trainer is created. With the shapes matched, the same
600k-decision run from random init scores 0.036 against `drill-throw-magnet` on
its own kit and 0.22-0.48 against the seven playing a foreign one, and PFSP puts
its largest weight on `drill-throw-magnet`. `v4_shape.ablate` is part of the
comparison: it is a writer bitmask, so a member trained with features ablated
would otherwise be fed features it has never seen.

The practical consequence is that the eight seeded specialists pin any run that
wants to train against them to their shape (`--encoder v4 --hidden 256
--aim-fourier 4 --v4-set-inner 64 --v4-ffn-dim 0 --v4-critic-dim 384
--v4-ablate 10`). A run that wants a different architecture trains with
`--opponents self` and joins the league by being rated against it afterwards,
which is what conditions are for.

## Why both Elo and Nash

Elo assumes strength is one number, which is only true when the payoff is
roughly transitive — and a league exists to manufacture non-transitivity. On
pure rock-paper-scissors every member is 0 Elo, correctly; add three redundant
copies of one member and Bradley-Terry invents a 113-point gap between the other
two, purely from population composition. Nash averaging rates a member by how it
does against the equilibrium mixture, and moves by ~1e-9 under the same cloning.
`check` demonstrates both.

The single-population "Nash value" `n·A·nᵀ` is identically zero for any
antisymmetric `A`, so it is not a readout; `rate` prints it only as an identity.
The scalar that carries information is RPP between two *different* populations —
here, the learned members against the scripted/random anchors.

## State of the league, measured 2026-09-24

Condition `sim9f60ac37.lineups-8.map0-1-2.t7200.d9`, 18 rated members (9 scripted,
8 specialist, random), 153/153 pairs, 7344 games, 48 per pair.

**One scripted bot is the whole equilibrium.** `search-hard` takes Nash mass 1.000;
support is 1 of 18 members and entropy 0.00 of a possible 2.89 nats. Every other
member has negative Nash value, from -0.250 (`grapple-beam-rock`) to -0.458
(`random`). Elo agrees with non-overlapping intervals: `search-hard` 497 [461, 538]
against 261 [228, 296] for the best specialist. Pairwise it is not close either -
`search-hard`'s own draw-aware score rate is at or above 0.750 against all
seventeen opponents, best resistance `grapple-beam-rock` at 0.250, worst
`teleport-mine-tesla` at 0.062.

**Read this as an off-kit result, not a verdict on the specialists.** The condition
is `lineups-8` at `--mirror-share 1`: eight three-ability loadouts, both players
always holding the same one. Each specialist was trained on exactly one of those
eight, so it plays its own kit in 1 of 8 lineups and a foreign kit in 7 of 8.
`search-hard` is a rollout search bot - it plans through `bopl_rl_fork` with
whatever abilities it happens to hold - so it is the only member that is
loadout-agnostic by construction. A condition that is 7/8 off-kit for everyone
else is close to the best case a search bot could be handed. The honest statement
is that the league measures **generalist** strength and the specialists are weak
generalists, which is what they were trained to be.

Three things follow.

- **The first exploiter should still target `search-hard`**, because it is the only
  member the equilibrium plays, so nothing else can move the mixture. But the
  exploit to look for is a search weakness, not a kit counter.
- **Adding members will not help until one of them beats it.** A pool whose
  equilibrium is a single strategy gives PFSP nothing to mix, so opponent sampling
  collapses to that bot however many members are registered. This is the
  saturation Matthew has already seen once.
- **The specialists on their own kits are a separate condition, and it has now
  been measured** - see "The own-kit condition" below. Counts are never pooled
  across conditions, so it is additive to this section rather than a revision of
  it. The short version: on their own kits the specialists are strong, and the
  equilibrium is still a single member.

Scope: 3 maps of 124. The condition tag `map0-1-2` is the map set 0, 1, 2
(`registry.condition` joins the sorted ids). `league play --maps` writes a separate
condition per map set, so widening the map set is the cheapest way to learn whether
`search-hard` dominates everywhere or only here.

`duplicator-blink-blackhole` and `dash-missile-platform` sit near the bottom, which
is expected: both lost a boundary-leaving exploit to the 2026-09-23 fidelity fixes
and are pending retrain. Their ratings here were measured post-fix and are honest,
so the degeneracy is not an artifact of carrying them. It is off-kit position only:
both are mid-pack in the own-kit condition below.

## The own-kit condition, measured 2026-09-24

Condition `simf0ab99a8.lineups-own-kit.map0-1-2.t7200.d9`, 9 rated members (8
specialists, random), 36/36 pairs, 1728 games, 48 per pair, every game played on
the participant's own three abilities and asserted as such from `lineup_ids`.

```
member                           Elo       bootstrap 95%     Nash    score
drill-throw-magnet               592  [    517,    675]   +0.000    0.743
grapple-beam-rock                527  [    464,    601]   -0.187    0.658
roll-grenade-smoke               495  [    432,    575]   -0.094    0.613
dash-bow-platform                451  [    383,    524]   -0.187    0.549
dash-missile-platform            448  [    381,    529]   -0.271    0.546
duplicator-blink-blackhole       393  [    331,    468]   -0.167    0.466
web-mjolnir-freeze               370  [    311,    443]   -0.333    0.434
teleport-mine-tesla              366  [    301,    430]   -0.333    0.427
random                             0  [      0,      0]   -0.375    0.064
```

`score` is the member's own draw-aware score rate over all 384 of its games,
random included; 6.3% of games (109/1728) were draws. RPP of the eight against
the anchor is +0.4792, i.e. 0.979 in score units.

**Training works.** This is what the section above could not say. Every
specialist sits 366 to 592 Elo above `random` on its own kit, and random takes
0.064 against the field. Off-kit in `lineups-8` the best specialist was 261, so
the gap to 592 is large - but do not read it as a clean measurement of the cost of
playing a foreign kit. The two conditions differ in the simulator library as well
as the loadouts (`sim9f60ac37` against `simf0ab99a8`, which is post-fixtures
#9-#14), which is exactly why counts are never pooled across conditions. Both
scales are anchored at `random` = 0, so the comparison is suggestive and the
direction is not in doubt; the magnitude is not a number to quote.

**The equilibrium is still one member.** `drill-throw-magnet` takes Nash mass
1.000 with support 1 of 9. So the own-kit population is saturated the same way the
generalist condition is saturated by `search-hard` - a different member, the same
shape. A pool whose equilibrium is a single strategy gives PFSP nothing to mix
regardless of how many members are registered, and that holds here too. Growing
this pool means adding something that beats `drill-throw-magnet`, not adding more
specialists.

**The ordering is near-transitive and the middle is not resolved.** Nash and Elo
agree on the ends but not in between. `roll-grenade-smoke` has the second-best
Nash value (-0.094) on the third-best Elo, and `duplicator-blink-blackhole` the
third-best (-0.167) on the sixth-best Elo, because Nash reads resistance to the
equilibrium member and Elo reads average strength against everyone. A member that
is mediocre overall but holds up against `drill-throw-magnet` is exactly what this
pool needs and exactly what Elo hides. Elo intervals separate top from
bottom (592 [517, 675] against 366 [301, 430]) and overlap everywhere in the
middle, so the five members between 393 and 527 are one indistinguishable band at
48 games per pair. Deepening them is what `play`'s 15-85% gate will spend CPU on
next.

**What this condition cannot separate: kit from policy.** Each member is one
policy welded to one kit, so `drill-throw-magnet` on top means the
drill/throw/magnet *pairing* is strongest, and there is no way here to say whether
that is a better-trained policy or a stronger set of abilities. Splitting it needs
either one policy rated across several kits or several policies rated on one kit,
and neither exists. Read the table as a ranking of specialist-plus-kit units, and
do not use it to argue any ability is strong.

`duplicator-blink-blackhole` (393) and `dash-missile-platform` (448) are mid-pack
here rather than at the bottom, which is worth noting because they are the two
pending retrain after the 2026-09-23 fidelity fixes. Their off-kit position near
the bottom of the generalist condition is not reproduced on their own kits.

Scope is the same 3 maps of 124 as the section above, and the same caveat applies:
widening the map set is the cheapest way to learn whether this ordering is general
or local.

## Rating the specialists on their own kits: what it takes

This is the design of the condition above, kept because the trap in it is the
transferable part. Since 2026-09-24 it is not a mode but the league's only
condition, generalised past specialists: every member declares the kits it can
play (`registry.kits`: scripted bots 2,4,14 because their slot indices are written
for it, a checkpoint the kits it trained on, random none), each seat plays one of
its own member's kits, and the two seats may differ as in a real match. A member
with no kit takes its opponent's; a member with several spreads its games over
the combinations (`league/__main__.py` `matchup`). Scripted bots also skip space
maps (`registry.plays_map`). Games run to 10800 ticks so sudden death, which the
game starts at 7200, decides them; at the old 7200 limit 17.5% of policy-vs-policy
games in tournament-20260917 ended as timeouts. The committed lineup file is gone:
the rows are derived from the members, which are immutable.

`lineups-8` at `--mirror-share 1` puts every specialist off-kit in 7 of its 8
lineups. The measurement wanted is each specialist on *its own* kit against the
others on theirs, and the obvious way to build it silently inverts the result.

**The trap.** A loadout-set row is six ability ids: player 0's three then player
1's three (`evaluation_protocol.resolve_loadouts` validates `row[:3]` and
`row[3:6]`; `train/bopl_puffer.h:959` iterates `player<2, slot<3` across the
row). Pinning a row therefore fixes *the kit each side plays*, not the kit each
participant plays. And `eval/elo_ladder.pair_tasks` emits a `side=0` and a
`side=1` task for every lineup index (`:272-277`). So pinning one asymmetric row
`[A|B]` per pair hands A its opponent's kit in exactly half the games, and the
resulting rating is a blend of on-kit and off-kit play that looks like a clean
number. `--mirror-share` must be 0 here, unlike `lineups-8`.

**The construction that avoids it.** Two rows per unordered pair, `[A|B]` and
`[B|A]`, with the side chosen to match: the participant whose kit comes first in
the row always plays player 0. Side balance then comes from alternating which row
is used, not from swapping sides under a fixed row. For 8 members that is 56
rows and 28 pairs.

This needs a task builder that pairs a lineup index with a specific side, since
`pair_tasks` deliberately couples both sides to each lineup. That is a small,
contained change next to `pair_tasks` rather than anything in the environment -
`configure_loadouts` per task is already cheap and uncached (`elo_ladder.py:300-312`
records the measurement: 96 pinned games cost the same wall clock as 96 unpinned).

**What to check on the way out.** `lineup_ids` is recorded per result
(`elo_ladder.py:339`, `played[0]+played[1]`), so the run can prove after the fact
that every participant played its own three abilities in every game. Assert that
rather than trusting the construction - it is the one cheap check that would have
caught the trap. `league play --own-kit` does, per game, and aborts before writing
any counts.

Two things the build found that the design above did not anticipate.

- **The anchor has no kit, and Bradley-Terry needs the anchor in the condition.**
  Rating the 8 alone fails outright: `anchor 'random' is not among the 8 members
  being rated; Bradley-Terry has no scale without it`. A kitless member therefore
  takes *its opponent's* kit, which is the honest reading - a random policy has no
  kit it was trained for, so mirroring its opponent's is the neutral choice and
  keeps the row symmetric. That makes it 9 members and 72 rows, not 8 and 56.
- **Row order must come from the league, not the command line.** The generated
  file is compared against the committed one and a mismatch is refused, so
  ordering rows by `--members` token order would make the same member set produce
  two different files depending on how it was typed, and the second would be
  rejected. `own_kit_rows` walks league member order.

One reporting subtlety: `expected` counts distinct row *contents*, not row
indices. The anchor's rows are `[X|X]` for every pair it plays, so eight of them
are duplicate content and 72 rows resolve to 64 distinct ones. Counting indices
prints a false `<-- INCOMPLETE` on a run where every scheduled game did play.
