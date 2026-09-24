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

`sim<sha8>.<loadouts>.map<ids>.t<ticks>.d<delay>` — the simulator library, the
loadout distribution, the map set, the episode limit and the observation delay.
That is everything that can change who wins a game.

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

Condition `sim9f60ac37.lineups-8.map012.t7200.d9`, 18 rated members (9 scripted,
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
- **The specialists have never been rated on their own kits against each other.**
  That measurement does not exist, and it is the one that would say whether
  training works. It is a separate condition (per-kit lineup file), and counts are
  never pooled across conditions, so it is additive rather than a revision.

Scope: 3 maps of 124. The condition tag `map012` is the map set 0, 1, 2
(`registry.condition` joins the sorted ids). `league play --maps` writes a separate
condition per map set, so widening the map set is the cheapest way to learn whether
`search-hard` dominates everywhere or only here.

`duplicator-blink-blackhole` and `dash-missile-platform` sit near the bottom, which
is expected: both lost a boundary-leaving exploit to the 2026-09-23 fidelity fixes
and are pending retrain. Their ratings here were measured post-fix and are honest,
so the degeneracy is not an artifact of carrying them.

## Rating the specialists on their own kits: what it takes

The state-of-the-league section above measures generalist strength, not specialist
strength, because `lineups-8` at `--mirror-share 1` puts every specialist off-kit
in 7 of its 8 lineups. The missing measurement is each specialist on *its own*
kit against the others on theirs. The primitives for it already exist; the design
has one trap that will silently invert the result, so it is written down here
before anyone runs it.

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
caught the trap.
