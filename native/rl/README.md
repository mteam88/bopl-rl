# Bopl native reinforcement learning

Training uses the pinned PufferLib native CUDA trainer. Persistent Unity IL2CPP
processes run the reconstructed game; the C environment provides observations,
stepping, rewards and counters. This is separate from the partial Rust gameplay
port in the parent `native/` directory.

The current diagnostic uses two players with Dash, Bow and Platform, three stocks,
and configurable maps. The larger goal remains diverse abilities, maps and
adaptive opponents; this diagnostic does not establish all-ability competence.
Throughput experiments and their limits are recorded in the
[research log](autoresearch/notes.md) and [evidence directory](evidence/).
Earlier trainer implementations and removed reports remain in Git history.

## Observation and control contract

- One action every four 60 Hz simulation ticks: 15 decisions per second.
- Each action contains one of 256 stick encodings and one of 16 combinations of
  jump, dash, arrow and platform buttons. Buttons remain held between decisions.
- World observations are nine ticks old (150 ms). Sampling at phase 3 of a
  four-tick decision and retaining two samples gives exactly nine ticks of age.
  The reset observation fills the delay buffer until history exists.
- The previous action is provided immediately: a player knows what they pressed.
- All players' loadouts and cooldowns are public, including during invisibility.
- An invisible opponent's coordinates and body hierarchy are omitted only while
  its effective sprite alpha is below 0.2. Independently owned grenades, smoke,
  spark nodes and pickups remain visible, including disabled-but-alive colliders.
- Features describe players, collider geometry and motion, slime movement states,
  public cooldowns, arena-relative positions, round time and previous input. Exact
  structured state is more precise than human vision; reaction delay alone does
  not establish human-equivalent play.
- The policy has space for 64 nearby visible entities, 32 visible slimes and 24
  body-associated ability slots. The raw C state is unbounded by these policy
  capacities. Overflow truncates deterministically and is counted in both
  training and evaluation metrics.

The layout-v3 observation vector has 3,690 floats. Player order starts with self;
entity and slime observations use nearest-first ordering, except pickups are
force-inserted and ropes collapse to at most three rows. Runtime ability ids gather
from the shared `E[40,64]` table and never enter a projection as ordinal scalars.
Positions scale by 64, velocities by 256, and extreme values clip. Wire object ids
remain exact integers but do not enter the policy.

## Native trainer

`puffer/train.py` drives native recurrent PPO. Rollout collection, the minGRU
policy, loss, backward pass and optimizer run in native code. Python configures
the run, publishes checkpoints and metrics, and invokes evaluation between
updates. The linear encoder and layout-v3 set encoder share the native minGRU
and action/value heads. `--encoder linear|set` selects the encoder explicitly.
The set encoder independently max-pools player, ability-slot, entity and slime
rows, then fuses those four 64-wide pools with the flat prefix/self/action path.

Half the games use current-policy self-play. The other half pair a learner with
a frozen opponent from PufferLib's Elo-rated self-play pool
(`pufferlib/selfplay.py`, driven by `puffer/native_selfplay.py`). Each
historical game is pinned to one of the frozen banks (four by default) and draws
its side at match reset. A bank retires its opponent when the learner's win rate
against it reaches `--swap-winrate` over at least `--min-games` games, or after
`--opponent-timeout` learner decisions; the replacement is sampled from the pool
(never the five newest snapshots once the pool has six), weighted toward newer
entries. The pool grows every `--snapshot-interval` learner decisions and on
win-rate swaps while small; past `--pool-size` it evicts every other entry of
the older half, so the initial policy is never evicted and old opponents stay
represented. Raw snapshots live in `<run>/pool/` (PufferLib keeps evicted files
on disk). A swap loads the new weights in place as soon as every game of that
bank has started a match since the previous swap, so at most one in-flight match
per historical game sees its opponent change mid-match; that match is excluded
from the bank's win-rate and Elo statistics. Pool state is recorded per update
under `pool` in `metrics.jsonl` and as events in `pool.jsonl`.

Rewards are +1 for a win, -1 for a loss, and -0.5 for each player at timeout.
There is no kill, death or per-tick reward. The default cap is 7,200 simulation
ticks (two minutes), configurable with `--episode-ticks`. Entropy regularization
decays exponentially from 0.01, halving every 30M learner decisions until it
reaches the 0.0005 floor at about 129.7M decisions. `--entropy-half-life` sets
that interval independently of the training budget. Learning rate is constant
at 0.0003 by default (`--anneal-lr 0`). Optional `--anneal-lr 1` uses native
cosine decay toward `--min-lr-ratio` over the run budget. Evaluation reads reward settings
from the checkpoint metadata.

`--steps`, `--checkpoint-every` and `--eval-every` count **learner decisions**.
A rollout finishes before stopping, so budgets may overshoot by less than one
rollout. JSON `step` and checkpoint filenames count **physical player decisions**
for compatibility with Learning Lab. Both counters are explicit everywhere.
Inactive padded opponent rows count in neither. `--learner-passes` sets native
minibatches per update; minibatch size is the learner rollout size, and the
wrapper corrects the native replay ratio for padded rows.

Each CUDA buffer contains whole self-play/historical pairs. Worker counts must
be even and divisible by twice the buffer count. CUDA graph capture requires
at least one warmup update (`--cudagraphs 1`, the default); `-1` disables it.
Warmup counters and losses are discarded, initial weights are restored by the
native trainer, and recurrent state is reset before the first game decision.

## Checkpoints, evaluation and metrics

A checkpoint consists of immutable `.bin` float32 weights and a `.bin.json`
sidecar. The sidecar records architecture, dimensions, both decision counters,
configuration and a SHA256 digest. Publication validates size and finite weights
before exposing the final file. Evaluation validates both files and performs
native CPU inference with the same weight layout.

`config.json` records the selected game and native binding identities.
`metrics.jsonl` contains physical/learner throughput, policy/value loss, joint
and per-head entropy, entropy coefficient, learning rate, advantage standard
deviation before normalization, gradient norm, learner button rates, and
cumulative completed games by map. Behavior counters include unfinished episodes
and exclude frozen players. `status.json` reports live progress; `result.json`
records completion, a requested stop, or failure.

Diagnostic evaluations run synchronously between updates. Each suite includes
side-swapped seeds against random, initial and optional frozen policies, held-out
maps, and a self-play replay. Final series evaluation uses separate seeds.
Saved games include lossless `.rep`, physics `.json` and `.mind.json.gz` files;
the latter records per-decision values and action probabilities for playback.
Use `--no-eval` for a throughput comparison that excludes evaluation work.

Learning Lab still consumes these run directories: learning curves, frozen
opponent results, truncation rates, behavior, replay playback and policy values.
The video worker is separate so rendering can be scheduled around training.

## Commands

Run from the Unity project root, the directory containing `Assets/` and `native/`.

```sh
python3 native/rl/puffer/setup.py
native/artifacts/rl/venv/bin/python native/rl/puffer/build.py --device cuda
native/artifacts/rl/venv/bin/python native/rl/research/native_policy.py --build
```

Select the exact reconstructed player for the experiment; its hashes are saved
in the run configuration. The sampled transport, spare game processes and
parallel startup are always enabled. Headless rendering work is skipped when
Unity uses its null graphics device; replay video keeps normal rendering.

```sh
native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
  --output native/artifacts/rl/native-smoke \
  --game Builds/LinuxIL2CPP/BoplBattle.x86_64 \
  --encoder set --workers 16 --banks 4 --steps 100000 \
  --training-maps 0,1,2 --heldout-maps 3,4 --episode-ticks 7200
```

To declare a sequential series, use `research/series.py prepare` with an explicit
native `--baseline`, output, learner budget, seeds, encoder and maps. Preparation
records source, runtime, baseline and frozen-opponent identities. Run the emitted
plan with `research/series.py run path/to/plan.json`. A changed identity requires
a new plan; incomplete runs are not silently resumed.

The native frozen-opponent manifest uses schema 2 and entries with `name`, `path`,
`sha256` and `metadata_sha256`. Relative paths resolve from the manifest directory.
Old Torch checkpoints and old series plans are not native checkpoints or plans.
Historical artifacts remain available as evidence; they are not converted by the
training command.

```sh
native/artifacts/rl/venv/bin/python native/rl/research/serve.py \
  --host 127.0.0.1 --port 8876 \
  --root native/artifacts/rl \
  --pool native/artifacts/rl/sweeps/anchors/pool-v6/pool.json \
  --human native/artifacts/rl/human/mac
native/artifacts/rl/venv/bin/python native/rl/research/watch_videos.py \
  --run native/artifacts/rl/native-smoke
```

`--root` auto-discovers every run (a directory with `metrics.jsonl` or `checkpoints/`). `--run-root NAME=PATH` and `--run-root-glob NAME=GLOB` remain for isolated workspaces; they are not required to list sweep trials. `--pool` is the **primary frozen-pool label** (sweep-objective IQM, run-page side cards), not the joint scale; if omitted, the dashboard prefers pool v6 when `sweeps/anchors/pool-v6/pool.json` exists. `--human` defaults to `human/mac`. Views: `/` all runs, `/ladder` joint Elo, `/lineups` runs × lineups, `/run/<name>`, `/sweep/<name>`. Copy `research/start-dashboard.example.sh` over the 8876 launcher to drop the old glob list.

Every run carries its loadout selection (`config.json["loadouts"]`, mirrored in each
evaluation's `loadout_selection`), so the board and the sweep table badge it —
`2·4·14` for a fixed loadout, `set lineups-8 ×8 (mirror 100%)` for a lineup file,
`pool 36 (mirror 50%)` for a random draw — and the board filters on the mode. A run
older than the multi-ability work has no block and is read as the production
2,4,14. The run page's **Loadouts** section is one line for a fixed run and opens
for a set or pool run into (a) a per-ability table over the last `LAST_K`
evaluations — activations per game the ability was actually drawn in, entity rows,
dropped rows (overflowing observation budget), and the abilities that were
available and never pressed; and (b) a per-lineup table: games, win rate vs random
/ vs scripted / vs frozen, timeout rate, the per-lineup frozen-pool rating when
`reference_pool.py rate-runs --loadout-set` has written one under
`<run>/reference/<pool_sha8>/loadout-<loadout_sha8>/`, and a win-rate sparkline per
lineup over the whole curve. `/lineups` is the same numbers as a runs × lineups
matrix for the runs whose config declares a selection. **The scripted bots press
ability slots** (slot 0 as Dash, 1 as Bow, 2 as Platform — `bopl_bots.h`), so off
2,4,14 they are degraded and their win rate is not comparable across lineups; the
UI labels that column and the 2,4,14 row stays marked as the one comparable to the
pool score. The aggregation is `research/lineup_report.py` (also a CLI:
`lineup_report.py --run <dir> [--last-k K] [--json]`), digested once per
`result.json` and cached on the evaluation stamps, so a live run costs one parse
per new evaluation. `research/check_dashboard_api.py` and
`research/test_dashboard_loadouts.py` cover the payloads.

A short smoke verifies execution and artifact compatibility. It does not establish
learning strength or whole-game competence. Report measured learner decisions,
completed matches and evaluation results separately from throughput and compile
success. Qualification and long-run results belong in the experiment evidence.

## Hyperparameter / reward sweeps

`sweep/driver.py` runs many short `train.py` trials under PufferLib's Protein
(cost-aware Bayesian search over a Pareto front of cost vs score) instead of
one long run at a time. One TOML per sweep declares fixed flags, swept
parameters as PufferLib spaces, and the scorer's settings — an external
yardstick (Elo against a frozen reference pool), never the self-play win
rate. Trials are ordinary run directories
under `native/artifacts/rl/sweeps/<name>/`, launched in their own systemd
scopes with a memory cap and a wall cap; the driver's `sweep.json` makes the
whole sweep kill-and-resume safe, and `summary.md`/`launch-best.sh` are
regenerated after every trial. The dashboard discovers those trial directories
under `--root` and shows each trial's labelled frozen-pool IQM plus its joint
Elo; the sweep page's score column stays the objective written in `sweep.json`.
See `sweep/README.md`.

```sh
native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
  --config native/rl/sweep/configs/smoke.toml --dry-run 3
```
