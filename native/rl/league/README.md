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
- `kind` — `main`, `exploiter`, `specialist`, `scripted`, `random`.
- `parent` — the member this was forked or reset from; `null` for random init.
  Every exploiter here is `null` on purpose: exploiters start from random
  weights, not from a fork of the main, so they are free to find combinations
  the main's representation cannot reach.
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

`a` and `b` are member *names* (unique and permanent); the reader folds a
`(b, a)` row into `(a, b)` with wins and losses swapped, and sums duplicate
rows, so one `play` session appends one row per pair and the file stays small.

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
final player flags via `research/elo_ladder.play_games`. The legacy `win` field
counted a mutual death as a win for whichever side wrote the report and is never
read. Games are always side-swapped pairs on fixed seeds.

## Commands

```
python native/rl/league seed     # write league.json: specialists, scripted roster, anchor
python native/rl/league play     # play matchups, append counts   (the only verb that costs CPU)
python native/rl/league rate     # Bradley-Terry Elo, Nash averaging, RPP
python native/rl/league status   # members, matrix coverage, saturation
python native/rl/league add      # append one immutable member
python native/rl/league check    # prove the estimators on matrices with known answers
```

`play` gates matchups it has already decided: a pair whose whole confidence
interval sits outside 15–85% is skipped as uninformative, which is where the
throughput goes. Re-running `play` deepens only what is still in doubt.

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
