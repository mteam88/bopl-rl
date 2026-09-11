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
- Other invisible players' coordinates are omitted from player, body and entity
  features before distance sorting and counting. The current filter conservatively
  omits all their owned physics entities. This requires further visibility review
  before an experiment including invisibility or smoke.
- Features describe players, collider geometry and motion, slime movement states,
  public cooldowns, arena-relative positions, round time and previous input. Exact
  structured state is more precise than human vision; reaction delay alone does
  not establish human-equivalent play.
- The policy has space for 96 nearby visible entities and 12 visible slimes. The
  raw C state is unbounded by these policy capacities. Overflow is counted and
  stops the research runner rather than silently producing a successful run.

The observation vector currently has 3,257 floats. Player order starts with self;
entity and slime observations use nearest-first ordering. Positions scale by 64,
velocities by 256, and extreme values clip. No Unity instance IDs enter the policy.

## Native trainer

`puffer/train.py` drives native recurrent PPO. Rollout collection, the minGRU
policy, loss, backward pass and optimizer run in native code. Python configures
the run, publishes checkpoints and metrics, and invokes evaluation between
updates. The linear encoder and entity/slime set encoder share the native minGRU
and action/value heads. `--encoder linear|set` selects the encoder explicitly.
The set encoder pools visible entity and slime features with shared weights;
it does not depend on their slot order.

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
  --root native/artifacts/rl --port 8876
native/artifacts/rl/venv/bin/python native/rl/research/watch_videos.py \
  --run native/artifacts/rl/native-smoke
```

A short smoke verifies execution and artifact compatibility. It does not establish
learning strength or whole-game competence. Report measured learner decisions,
completed matches and evaluation results separately from throughput and compile
success. Qualification and long-run results belong in the experiment evidence.

## Hyperparameter / reward sweeps

`sweep/driver.py` runs many short `train.py` trials under PufferLib's Protein
(cost-aware Bayesian search over a Pareto front of cost vs score) instead of
one long run at a time. One TOML per sweep declares fixed flags, swept
parameters as PufferLib spaces, and a scorer — an external yardstick (Elo
against a frozen reference pool, a behaviour rate, or a head-to-head match),
never the self-play win rate. Trials are ordinary run directories
under `native/artifacts/rl/sweeps/<name>/`, launched in their own systemd
scopes with a memory cap and a wall cap; the driver's `sweep.json` makes the
whole sweep kill-and-resume safe, and `summary.md`/`launch-best.sh` are
regenerated after every trial. The dashboard lists trials via
`serve.py --run-root-glob 'name=native/artifacts/rl/sweeps/name/[0-9]*'`.
See `sweep/README.md`.

```sh
native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
  --config native/rl/sweep/configs/smoke.toml --dry-run 3
```
