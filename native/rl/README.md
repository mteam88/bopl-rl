# Bopl reinforcement-learning research

Completed runs and measured outcomes: [experiment log](EXPERIMENTS.md).

This environment runs the actual reconstructed Unity game in persistent headless
processes, controlled through the C API in `bopl_rl.h`. It is separate from the
partial Rust gameplay port in the parent `native/` directory. The game engine
provides gameplay; the C layer supplies observations, action stepping and rewards.

The first research experiment uses **two players, dash + arrow (Bow) + platform,
three stocks, and map 0**. It asks whether a recurrent policy can learn to survive
and win with delayed information. Evaluation also uses maps 1 and 2. This is a
bounded training experiment, not a claim of complete all-ability compatibility.

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

## Training and evaluation

`research/train.py` implements recurrent PPO with a 128-unit GRU, using the pinned
PufferLib C vector runner for concurrent environment stepping and CUDA buffers.
The rollout records rewards after each action, carries recurrent memory between
rollouts, clears it at episode boundaries, and preserves segment-start memory
and internal reset masks during optimization.

Half the games use current-policy self-play. The other half use one current player
against a historical checkpoint. Historical actions are excluded from PPO losses.
A match keeps the same opponent until it ends; the pool retains four recent
checkpoints, plus any older version still in an active match.

Rewards are +1/-1 for a round win/loss and +0.1/-0.1 per kill/death. There is no
per-tick survival reward. A 3,600-tick match limit is part of the finite-horizon
task: timeout ends the episode with no winner bonus and a terminal value of zero.

Each run writes its configuration, source hashes, game assembly hash, checkpoints
(with optimizer state), metrics, evaluation reports, and replays. `--load` is a
warm start of weights/optimizer in fresh environments; it is not a bit-exact
resumption of ongoing matches or the historical opponent pool.

Evaluation uses identical seeds at each checkpoint and swaps player sides. It
reports wins, timeouts, kills, deaths and duration against random buttons and the
initial policy, plus performance on two untrained maps. The default eight games
per training-map opponent are a fast diagnostic, with wide Wilson intervals.
A promising policy needs a larger, independently seeded tournament before a
strength claim. Self-play's aggregate ~50% win rate is not evidence of improvement.

Every evaluation exports a lossless `.rep` plus a full physics-state playback.
A separate video worker renders native game playback. If rendering falls behind,
it keeps the initial and most recent checkpoints rather than delaying training.
Native and physics replays remain available for every completed evaluation.

## Commands

Run from the Unity project root (the directory containing `Assets/` and `native/`).

```sh
python3 native/rl/puffer/setup.py
native/artifacts/rl/venv/bin/python native/rl/puffer/build.py --device cuda
make -C native/rl

native/artifacts/rl/venv/bin/python native/rl/research/preflight.py \
  --output native/artifacts/rl/research-checks

native/artifacts/rl/venv/bin/python native/rl/research/train.py \
  --output native/artifacts/rl/dap-001 --workers 16 --steps 1000000

python3 native/rl/research/serve.py --host 127.0.0.1 --port 8876
```

An output directory must be new. The normal training command requires CUDA and
runs evaluation and video capture. `--no-eval` is for short throughput checks;
`--no-video` retains evaluation and both downloadable replay formats.

After a C observation/binding-only change, `build.py --device cuda --binding-only`
reuses the existing CUDA object and relinks the environment. Do not use that flag
after changing PufferLib CUDA sources, compiler flags or the dependency revision.
Do not rebuild the game or binding while workers are using them.

The dashboard is read-only. It supports phone layouts, live graphs, checkpoint
selection, play/pause, scrubbing, speed selection, native video and `.rep` downloads.
When BB Connect is paired, use `bb connect expose 8876`. On Matthew's PowerSpec,
the current dashboard is served through a dedicated tailnet-only HTTPS route:

```sh
tailscale serve --bg --https=8876 http://127.0.0.1:8876
# Remove only this route when the dashboard is retired:
tailscale serve --https=8876 off
```

Native video capture requires `Xvfb`, `xdotool`, `ffmpeg` with `h264_nvenc`, and
`ffprobe`. It owns and cleans up only its own display, game and encoder processes.
Do not use the older root-level `record-demo.sh` during training: it kills all
Bopl processes by name.

For headless browser verification, start an isolated Chrome profile with
`--headless --password-store=basic --use-mock-keychain --remote-debugging-port=9224`,
then run `node native/rl/research/check_dashboard.mjs <dashboard-url>`.

## Evidence and limits

Preparation checks cover real-game delay/reset/loadout behavior, hidden-position
invariance with public cooldown changes, recurrent reset equivalence on CPU/CUDA,
exact checkpoint reload, finite loss and updated weights in short GPU jobs,
evaluation/replay export, and desktop/phone browser behavior.

`verify_replay.py --players 2 --ability-group 4 --ticks 900` compared 1,140 ticks
(including warmup) with ordinary playback and matched the audited gameplay
projection. That episode entered Bow 7 times, Dash 4 times, and Platform 2 times.
The projection excludes explicitly audited presentation fields listed in
`native/data/presentation-fields.json`; it is not an unqualified full-object hash.
Broader all-ability replay work remains separate and unfinished.

Performance tuning so far: persistent game processes, compiled C feature packing,
shared CUDA buffers, batched policy inference, passive OpenMP waiting, removal of
repeated loadout features, contiguous GRU weights, and reuse of unchanged CUDA
build objects. A matching 65,536-decision comparison measured median throughput of 3,252
decisions/s with eight workers and 4,490 with sixteen (38% higher). Report wall
throughput for the actual research run, including evaluation/video contention;
do not extrapolate the native fused smoke benchmark to this recurrent workload.

Collection profiling (2026-09-06, sixteen workers, `research/profile_collection.py`):
one vectorised step costs about 7.4 ms, of which the game/IPC step is 3.3 ms
median (5 ms mean; ~33 ms scene resets stall the OpenMP barrier), and the
remainder is serial GPU-side Python work. Games use about 0.4 cores each and a
single uncontended game needs 0.63 ms per four-tick decision, so the vector is
bound by two IPC round trips per decision, not by CPU. Two exact changes landed:
`Categorical(validate_args=False)` in `policy.distribution` (removes a per-call
GPU sync; samples and log-probabilities are identical) and `research/selfplay.py`,
which caches per-version opponent index tensors instead of rebuilding Python
index lists every step (`check_selfplay.py` verifies bit-exact equivalence).
Matched 65,536-decision `--no-eval --no-video` runs, alternating three times each
on an otherwise idle machine, measured 4,231–4,367 decisions/s before and
4,562–4,781 after (median 4,319 vs 4,773, about +10%); final weights and RNG
state were bit-identical across all six runs. Profiler numbers exclude
evaluation/video and the `--no-eval` runs exclude evaluation contention, so
expect lower wall throughput in a full research run.
