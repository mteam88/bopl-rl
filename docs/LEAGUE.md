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
- `loadouts` — the kits the member can legitimately play (`registry.kits`):
  `null` for any kit (random, and policies trained on a kit draw), `fixed` for
  one (a specialist's own kit; the scripted bots' 2,4,14, the only kit their
  slot indices mean anything on), or a `set` of kits. Scheduling takes every
  game from these, so a member is never handed a kit it cannot play.
- `added` — `{date, rule}`: when, and which rule admitted it.

Members are immutable. `league_sha256` covers the member list, and every
checkpoint's weights are re-hashed on load; either mismatch refuses the load.

## `payoff.jsonl`

One JSON object per line, append-only:

```
{"a": "<member id>", "b": "<member id>", "condition": "<key>",
 "wins": <a's wins>, "losses": <a's losses>, "draws": <draws>}
```

Aggregated counts only — never per-game rows. Bradley-Terry and Nash need only
counts, and a league that grows for months would otherwise commit millions of
lines to git permanently. Per-game detail is derived data and belongs under
`native/artifacts/`.

`a` and `b` are member *ids*, unique and permanent.

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

## The rating condition

`registry.rating_condition()` is the one condition ratings are fitted on:
this simulator, each seat on its own member's kit (`own-kit`), all 124 maps,
10800-tick games (the game starts sudden death at 120 s, so close games are
played out rather than cut off as draws) and the 9-tick observation delay.
Today that is `simf0ab99a8.own-kit.mapall-124.t10800.d9`; a new sim build gives
a new key and the league is re-played on it.

`league/games.py` is the one place a game's map and kits are chosen. Between two
members, seed pair i plays kit combination i (every kit of one against every kit
of the other; a member with no kits of its own takes its opponent's) and map i of
one fixed random order of the maps both can play (the scripted bots cannot play
space maps), both shifted by a per-matchup offset. `start` continues from the
pairs already recorded, since games are deterministic in their seed.

## Rating a checkpoint

`league candidate <checkpoint>` (`candidate.py`) measures a training run. The
checkpoint plays the members on the kits its run trained on (the checkpoint's
sidecar config), the members on theirs, under the rating condition; the
members' ratings are held fixed and the checkpoint's rating is the 1-D
Bradley-Terry MLE of its games (OpenAI Five's frozen reference rating). Games go
where they carry information: a first pass against every member, then rounds
against the members it has a 15-85% chance against. The interval is a
bootstrap over side-swapped seed pairs, and a run that trained on several kits
gets a rating per kit beside the overall one.

The result is `<run>/league/<checkpoint stem>.json`, with the games. Members'
ratings move as the league grows, so readers (`candidate.load`, `curve`) refit
the stored games on the current ratings. The trainer rates every evaluation
checkpoint with a small budget (`source: in-run`); the sweep scorer rates the
last three with the full one (`source: band`) and scores their IQM.

## Scoring

Draw-aware, always: a win is 1, a draw 0.5, a loss 0, computed from each game's
final player flags via `eval/elo_ladder.play_games`. The legacy `win` field
counted a mutual death as a win for whichever side wrote the report and is never
read. Games are always side-swapped pairs on fixed seeds.

## Commands

```
python league seed     # write league.json: specialists, scripted roster, anchor
python league play     # play matchups, append counts
python league candidate <checkpoint>   # rate a checkpoint against the members
python league rate     # Bradley-Terry Elo, Nash averaging, RPP
python league status   # members, matrix coverage, saturation
python league add      # append one immutable member
python league check    # prove the estimators on matrices with known answers
```

`play` gates matchups it has already decided: a pair whose whole confidence
interval sits outside 15–85% is skipped as uninformative, which is where the
throughput goes. Re-running `play` deepens only what is still in doubt.

There is exactly one rating scale in the repo: `eval/elo_ladder.py`'s
`bradley_terry_fit` (draw-aware, `random` anchored at 0 Elo, with intervals),
fitted on the rating condition. This package calls it and does not reimplement
it. The frozen reference pools that came before it (`sweeps/anchors/pool-v*`,
`reference_pool.py`) are gone; their trained members are league members.

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
would otherwise be fed features it has never seen. The exception is a writer
*revision* (`registry.WRITER_REVISIONS`, today bits 16 and 32, layouts v4.2 and
v4.3): it changes what some fields hold but not the row's shape or what a field
means, so it is masked out of the comparison. A v4.2 run (`--v4-ablate 24`) can
start from a v4.1 checkpoint and load v4.1 members, and each frozen bank writes
its member's own rows (`bank_kits.WriterBackend`, `bopl_bank_ablate`). Layout
v4.3 (bit 32, e.g. `--v4-ablate 48` on a v4.2 net) reads a Drill form's real
body velocity and heading in the body blocks, casts rays from inside the
ground to where they leave it, and no longer also draws a floating platform as
an object circle (train/bopl_obs_v4.h, bopl_rows_v4.h). It needs the sim's Drill
form export (`BOPL_RL_FORM_BODY=1`, which train.py and `native_policy` set
when a run or a loaded member has the bit).

The practical consequence is that the eight seeded specialists pin any run that
wants to train against them to their shape (`--encoder v4 --hidden 256
--aim-fourier 4 --v4-set-inner 64 --v4-ffn-dim 0 --v4-critic-dim 384
--v4-ablate 10`). A run that wants a different architecture trains with
`--opponents self` and joins the league by being rated against it afterwards,
which is what conditions are for.

## Kit pinning: frozen opponents on their own kits

A frozen bank normally plays whatever kit the run deals its seat. An exploiter or
specialist built for a handful of kits (`anticamp-camper-1000m`: four Dash/Platform
kits) is then handed the learner's draw from `--loadout-pool all` and plays
off-kit, so main never meets the thing it was built to punish. With

```
--pinned-opponents anticamp-camper-1000m,anticamp-punisher-4000m
--bank-groups all,all,all,pinned
```

a bank that loads one of those members deals its frozen seat one of the member's
`league.json` kits (`registry.kits`, i.e. its `loadouts`: a `fixed` kit or a
`set`'s kits) in every match, on both sides. The learner keeps its own draw (the
pool, a set's learner half, or the fixed kit): the point is to test main's
general play against the specialty, and pinning the learner too would turn the
matchup into a fixed lineup main does not otherwise train on. The C side
(`bopl_bank_kit_rows`, `train/bopl_puffer.h`) keeps a row table per bank,
`[member kit | learner kit]` with a learner half of `0,0,0` meaning "the run's own
draw"; a nonzero learner half is supported in C but nothing sets one yet. Rows
are installed between rollouts by `bank_kits.PinnedBackend.load_frozen_bank`
just before the weights, under a generation counter, so a setup drawn before a
swap (the spare world deals one episode ahead) is re-dealt at reset rather than
played with the previous member's kit.

The `pinned` bank group draws only pinned members, and while a run has a
`pinned` bank no other group draws them, so their share of games is that bank's
share: one bank of four at `--historical-every 2` is 12.5% of all games, close to
AlphaStar's 15% for main exploiters. PFSP inside the bank is the run's own
(`--pfsp-p`). Without a `pinned` bank, pinned members are ordinary candidates of
every group and are only pinned when drawn.

`--pinned-opponents auto` (alone) pins every `--opponents` member that has kits
in `league.json`: exploiters and specialists, but also every main trained on a
lineup (`g111-*`, `l1-*`: their 15-16 g111 kits) and `v4-lite-*` (2,4,14).
Members trained on `--loadout-pool all` have no kits and play the run's draw.

**The `forgotten` bank group** (AlphaStar's 15% of main-exploiter and
"forgotten" past-main games) draws every `kind: exploiter` member (main and
league exploiters alike) plus any member the learner currently scores under 0.5
against (`FORGOTTEN` in `native_selfplay.py`), falling back to `league` and then
`all` when that is empty. A loaded opponent is not swapped out just because the
learner starts beating it: the bank's staleness check uses the whole pool.

**A scripted bot can share the bank.** `--scripted-roster rusher-hard
--scripted-banks 1` puts the bot on the last bank; at a fixed
`--scripted-share-min = --scripted-share-max` it takes that fraction of the
bank's matches and the pinned members take the rest. Under pinning the bot plays
its own kit (2,4,14, the only one `bopl_bots.h` understands) whatever the run
deals, and never a space map: a space-map match goes to the bank's frozen
member, decided with the setup. `rusher-hard` has no payoff rows, so no league
rating; `--scripted-elo rusher-hard=0` supplies the starting Elo the roster
needs (with one entry at a fixed share the number only labels telemetry).

Proof: `make check-bank-kits` (train/check_bank_kits.py) steps the real binding
on both reset paths and asserts that every pinned match held one of its member's
kits on both sides, across a mid-run swap; that bot matches hold 2,4,14 and no
space map; and that without the flag, or with it and no rows installed, every
match is dealt exactly as before (also checked once against a binding built
before this feature, `--golden`). In training, `metrics.jsonl` carries
`bank_kits` per bank (member, pinned matches, bot matches, matches from before
the last swap, `off_kit`, which must stay 0) and the console prints it.

To add an exploiter later: `league add` it with its kits in `loadouts` (a run
trained with `--loadout-set ... --loadout-learner-first` declares the learner
halves), then name it in `--pinned-opponents` (or rely on `auto`); a
`--resume` picks it up.

## League loop l2

`train/league_loops/l2.toml` runs `train/league_loop.py` as an AlphaStar-style
league around f5 (`final5-18666m`); the loop's docstring has the design. In short:

- **main**: continues f5 with f5's config (same reward, no style), 2e9 steps per
  turn. Half its games are self-play (the vector's learner-vs-learner
  environments; 35% would need a C change, since `--historical-every 2` fixes
  historical environments at 1/2). The other half are over four banks:
  `all,all,all,forgotten` with `--pfsp-p 3` (hard), the league re-read every
  turn (`seed_members = "league"`), every member with kits pinned (`auto`), and
  rusher-hard on 20% of the forgotten bank's matches. Each snapshot is
  `league add`ed and gated with `league/all_combos.py` against random, f5 and
  A20 (`gates/<name>.jsonl`, one line per turn in `summary.md`).
- **main exploiter**: the next recipe from `train/league_loops/recipes.toml`
  (seeded with the anticamp camper and punisher; append a `[[recipe]]` to add
  one, read at every reset). Warm-starts from the newest main with
  `--loadout-set <recipe> --loadout-learner-first`, plays only that main, 1e9
  steps per turn. After each turn `eval/exploit_eval.py` plays the exploiter and
  the main itself (the control) on the same seeds and kits against the main. If
  the paired difference's 95% interval clears `exploit_margin` it is added with
  its kits (and so pinned in every later main turn) and reset; at 2e9 it resets
  unregistered.
- **league exploiter** (every third exploiter turn): random init, `--loadout-pool
  all`, PFSP over the whole league; joins when its pooled score against
  `league_members` sampled members clears 0.5 with 95% confidence; 4e9 timeout.

The learning rate is constant (`--anneal-lr 0`): train.py's cosine restarts at
every `--resume`, so it cannot span turns. Members are append-only. State is in
`<dir>/state.json`, `turns.jsonl` and `summary.md`; stopping and restarting the
loop's unit adopts a running turn or resumes it. Evaluations are CPU jobs, one
at a time, in their own units (`bopl-<loop>-job-*`) at `MemoryHigh`
`[evals] memory_high`, overlapping the next GPU turn.

```
systemd-run --user --unit=bopl-l2 --collect -p MemoryHigh=12G \
  -p WorkingDirectory=$PWD --setenv=BOPL_RL_BACKEND=rust \
  $PWD/native/artifacts/rl/venv/bin/python -u train/league_loop.py train/league_loops/l2.toml
```

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

## State of the league, measured 2026-10-06

The rating condition `sim32b8446a.own-kit.mapall-124.t10800.d9`, the sim after
the sim-fidelity merge (1e08217c): 58 members (18 v4 and 4 v3 mains, 8
specialists, 27 scripted bots, random), every pair played, 24 games each, 39,672
games in 121 min. 540 games (1.4%) were draws.

```
member                  Elo   95%            member                  Elo   95%
v4-full-2240m           495  [464, 527]      search-easy             435  [406, 467]
dash-bow-platform       480  [450, 512]      evasive-v2-hard         371  [341, 404]
v4-full-2313m           479  [449, 511]      v3b-041-final           355  [326, 387]
v4-aim-only-2313m       476  [444, 506]      drill-throw-magnet      320  [286, 354]
search-medium           466  [437, 497]      dash-missile-platform   280  [252, 310]
v4-lite-1920m           465  [436, 497]      roll-grenade-smoke      248  [220, 276]
grapple-beam-rock       444  [411, 477]      teleport-mine-tesla     212  [184, 242]
search-hard             442  [408, 473]      duplicator-blink-bh     209  [182, 239]
                                             web-mjolnir-freeze      192  [159, 226]
                                             random                    0
```

(`league rate` prints all 58.) Against the same round robin on the previous sim
(`simf0ab99a8`, 2026-09-24) the mean move is 5 Elo and the largest 16, all
inside the intervals: the fidelity fixes did not reorder the league. Four things
in it.

- **The v4 mains and the search bots are one band.** From `v4-full-2240m` at
  495 down to `search-easy` at 435 the intervals overlap throughout: the best
  trained policies are level with the search bots, not past them.
- **Every other specialist is below the v4 mains, most well below.** They were
  trained on maps 0-2 only, and here they play all 124. `dash-bow-platform`
  sits among the v4 mains on their own kit, which says the kit, not the
  training, sets most of the gap. `drill-throw-magnet`, first by 65 Elo when
  rated on maps 0-2, is 320 here: its strength was largely those three maps.
- **The Nash equilibrium is five members**: `grapple-beam-rock` (0.423),
  `dash-bow-platform` (0.204), `search-easy` (0.183), `v4-aim-only-2313m`
  (0.148) and `search-hard` (0.042). On the previous sim it was two,
  `v4-aim-only-2313m` and `grapple-beam-rock`. RPP of the learned members
  against the 28 scripted and random anchors is +0.166 (their equilibrium beats
  the anchors' at 0.666; 0.689 on the previous sim).
- **A checkpoint rated by `league candidate` lands where the round robin puts
  it.** Checked on the previous sim: `drill-throw-magnet`'s checkpoint, rated as
  a candidate on 114 games, refit to 327 [268, 383] against its member rating
  of 327.

`league play` sends pairs to the workers 40 at a time (`PLAY_BATCH`). Played one
pair at a time, each pair ended with its longest game running alone, and the
same round robin took 4.6 h on 9 workers. Games are fixed by their seeds, so
batching changes no result.

Map 114 (`espace 1`) is a hazard, not a defect: a moon orbits each spawn planet
and its first pass, about half a second in, crushes a player that stays on top of
the planet. The current members, none trained on space maps, often walk into it.

## Rating the specialists on their own kits: what it takes

This is how the own-kit rating condition came about, kept because the trap in it is the
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
