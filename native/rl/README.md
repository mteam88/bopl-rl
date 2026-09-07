# Bopl reinforcement-learning research

Completed runs and measured outcomes: [experiment log](EXPERIMENTS.md).

Latest comparison: [A/B continuation and direct crossplay](evidence/opponent-extension-001/REPORT.md).

Long-term goal and initial skill diagnostic: [Bow curriculum](BOW_CURRICULUM.md).

Human demonstration extraction, supervised GRU fits and replay diagnostics:
[imitation workflow](IMITATION.md).

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
The pilot uses the game's raw kill counter, which also credits self-kills. Thus
the small kill/death terms cancel for a suicide. The controlled three-seed series
preserves this behavior; excluding self-kill bonuses is a separate reward ablation.

Rollout collection is pipelined (`research/vector.py`, `native/rl/puffer/binding.c`):
the workers are split into contiguous groups, each with its own C thread and CUDA
stream, so one group's games tick while the policy runs inference for the other.
Observation age (9 ticks), reward alignment, recurrent resets, historical-opponent
ownership and on-policy update boundaries are unchanged; `--pipeline-serial` waits
for each group immediately and reproduces the overlapped run bit-exactly, and
`--pipeline-groups 1` reproduces the previous synchronous loop bit-exactly.
Rollout inference for the learner and every pooled historical policy replays
captured CUDA graphs (`research/inference.py`, `--cuda-graphs 0` for eager) with
identical random draws and recurrent state. Each decision is one game round trip:
the bridge's sampled step (op 6) runs the 3-tick and 1-tick phases together and
returns the intermediate state used for observations plus the final state and
telemetry. `BOPL_SPLIT_STEP=1` forces the old two-round-trip path (test oracle);
`step_mode` in `config.json`/metrics records which path a run actually used, and
training refuses to run on an old game build that would silently fall back.
Defaults are two groups (for even worker counts) and CUDA graphs on; see
[Evidence and limits](#evidence-and-limits) for the measurements behind them.

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

The longer series keeps historical checkpoints at the same cadence as the pilot,
but evaluates at 1M, 3M and 10M decisions. These milestones round up to a completed
rollout. `--eval-at` changes only evaluation scheduling. It does not change the
opponent pool, PPO updates, or random-number consumption. The three fresh training
seeds share one frozen baseline opponent. Final evaluation uses separate seeds
that are never used in the checkpoint diagnostics.

## Behavioral measurements

Protocol/ABI version 2 adds episode event counters after the existing state.
The policy observation remains 3,257 floats; no telemetry enters its input or
reward. Counters exclude warmup and track actual ability entries, arrows released,
accepted damaging arrow collisions, and deaths by the game's credited killer.
Environment deaths include falls and hazards; this is credit attribution, not a
claim that an opponent could not have caused the fall. Self-kills are separate.
Arrow hits exclude the owner and count accepted damaging collisions, not unique
arrows or every geometric contact. Ability entry does not imply a useful action.

Each native vector worker writes per-decision deltas to a separate host buffer.
The trainer selects exactly the decisions used for learning, including the final
decision before a reset. Historical opponents are excluded. The dashboard shows
rolling 20-update event rates per simulated player-minute, a breakdown of the
unchanged reward per 1,000 learner decisions, and recent learner-only button rates.
The legacy pilot is labeled separately because it lacks these counters.

`research/check_telemetry.py --baseline <old-project-root>` checks event/death
conservation and reset behavior, then compares exact fixed-point gameplay state,
every delayed observation, and lossless replay hashes against the old build.

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

Throughput and pipeline correctness checks (GPU; the game must be rebuilt after
`RlBridge.cs` changes, see `native/README.md`):

```sh
native/artifacts/rl/sampled_check Builds/Linux/BoplBattle.x86_64 \
  native/fixtures/movement.rep /tmp/sampled_check.log 1     # exact one-trip vs two-trip stepping
native/artifacts/rl/venv/bin/python native/rl/research/check_pipeline.py \
  --output native/artifacts/rl/throughput/pipeline-check    # transport, serial/async, alignment, graphs
native/artifacts/rl/venv/bin/python native/rl/research/bench_throughput.py \
  --output native/artifacts/rl/throughput/bench --workers 16 --repeats 3 --decisions 393216 \
  --arms "sync=--pipeline-groups 1 --cuda-graphs 0,default="
```

For a declared three-seed series, prepare after building and checking the final
implementation, then execute the saved plan. The runner checks source/binary
identity before each seed, runs seeds sequentially, and stops on a failed run.
It does not silently resume an interrupted training environment.

```sh
native/artifacts/rl/venv/bin/python native/rl/research/series.py prepare \
  --output native/artifacts/rl/dap-series-001 --steps 10000000 \
  --workers 16 --seeds 73,173,273 \
  --baseline native/artifacts/rl/dap-001/checkpoints/policy-000000000.pt
native/artifacts/rl/venv/bin/python native/rl/research/series.py run \
  native/artifacts/rl/dap-series-001/plan.json
```

The series dashboard opens with `?series=dap-series-001` and follows the active
seed. Diagnostic evaluation uses 64 games per training-map opponent and 16 per
untrained map. Final evaluation uses 256 and 64 respectively, with fresh seed
pairs. Report individual training-seed results; Wilson intervals are descriptive
and do not adjust for dependence within side-swapped pairs.

The multi-map extension selects an explicit set at each episode reset. A single
map retains the original reset RNG sequence. `--training-maps 0,1,2` samples only
those IDs; the game has a larger map catalog, so this differs from the low-level
`random_maps` option that samples every map. Completed training matches by map
are recorded in each metrics row. Map IDs in `--heldout-maps` must not appear in
training; the dashboard labels maps according to the saved configuration.
`research/check_map_contracts.py --output <new-artifact-directory>` checks native
selection, stable log keys, and immediate failure on observation overflow without
starting a game or GPU. Python schedule/identity tests run with
`python3 -m unittest discover -s native/rl/research -p test_evaluation_protocol.py`.

Frozen trained opponents are described by a schema-1 JSON manifest containing
`name`, `path`, and `sha256` for each checkpoint. Relative paths resolve from the
manifest. The series copies and checks their identities, and evaluators reject
changed weights or a conflicting existing evaluation. These opponents enter
evaluation only; training retains its current/recent-checkpoint self-play mix.

After qualifying the selected maps with `research/check_maps.py`, prepare the
next experiment using the three final policies from `dap-series-001`:

```sh
native/artifacts/rl/venv/bin/python native/rl/research/check_maps.py \
  --output native/artifacts/rl/multimap-checks --maps 0,1,2,3,4
native/artifacts/rl/venv/bin/python native/rl/research/series.py prepare \
  --output native/artifacts/rl/dap-multimap-001 --steps 10000000 \
  --workers 16 --seeds 73,173,273 --training-maps 0,1,2 --heldout-maps 3,4 \
  --pipeline-groups 2 --cuda-graphs 1 \
  --baseline native/artifacts/rl/dap-001/checkpoints/policy-000000000.pt \
  --opponents native/rl/evidence/frozen-dap-series-001.json \
  --diagnostic-games 32 --diagnostic-frozen-games 16 \
  --final-games 256 --final-frozen-games 128
```

This schedule has 369 games per diagnostic and 2,817 per final suite. It evaluates
each frozen policy on every training map, with separate random-play checks on
the two unseen maps. Each suite includes one self-play game for a native video.
Series preparation also records `--pipeline-groups`, `--cuda-graphs` and
`--split-step` explicitly (defaults `1`, `0`, `0`), so later trainer defaults or
the parent shell cannot change a declared run. The series disables test jitter.
The series samples GPU activity, memory, power and host CPU load once per second
in `resource-samples.jsonl`. These measurements cover the whole device and host;
the training phase includes startup, checkpoint evaluation and video work.

After the series finishes, generate a portable JSON summary and PNG/SVG figures
with a separate Python environment containing NumPy and Matplotlib:

```sh
python3 native/rl/reports/summarize_series.py \
  --series native/artifacts/rl/dap-series-001 \
  --output native/artifacts/rl/dap-series-001/report
```

This reporter checks the declared final game counts and per-game outcomes. Its
final error bars resample whole side-swapped seed pairs, with 5,000 bootstrap
draws. They describe evaluation uncertainty for each fixed policy; they do not
pool training seeds or establish performance against human opponents.
`--allow-partial` creates an explicitly labeled preview while a series is running.

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

Throughput work (2026-09-06, [evidence/throughput/summary.json](evidence/throughput/summary.json),
raw logs under `native/artifacts/rl/throughput/`). Worker counts 16/24/32 were
benchmarked with sampled GPU utilization/power and CPU load; 32 workers is about
9% faster than 16 in loop throughput but saturates the six cores and changes the
PPO batch, so 16 stays the series contract. The three runtime changes were then
measured in matched `--no-eval --no-video` runs of 393,216 decisions (96 updates,
crossing three checkpoint snapshots so the pool reaches four versions and graph
capture for new snapshots is paid), three repeats each, at 16 workers:

| arm | loop decisions/s | active (incl. snapshots) | wall (incl. 36 s startup) | GPU util | busy cores |
|---|---|---|---|---|---|
| synchronous loop (`--pipeline-groups 1 --cuda-graphs 0`) | 4,455 | 4,448 | 3,131 | 8.8% | 5.5 |
| default (2 groups, CUDA graphs) | 5,547 | 5,506 | 3,605 | 15.6% | 6.1 |
| default with `--pipeline-serial` (oracle) | 3,986 | 3,965 | 2,873 | 12.4% | 4.3 |

That is +24% active and +15% wall throughput over the synchronous loop. All three
serial/async pairs match exactly across every update (rewards, matches, losses,
KL, gradients, behavior, pool state). Overlap alone raises active throughput
by 39% against the identical grouped serial control; the 24% gain against the
synchronous baseline includes both grouping and CUDA graphs. The 98,304-decision
screen of groups 1/2/4 with and without graphs (two repeats) put 2 groups with graphs
first at 16 workers; 4 groups without graphs was slower than synchronous. Graph
capture and checkpoint snapshots cost under 0.05 s per run. GPU utilization stays
low because the loop is bound by game IPC and per-step launch overhead, and loop
throughput falls in every arm as the historical pool grows from one to four versions.

Correctness: `check_pipeline.py` passed all phases (transport ordering with 2/4
groups and injected jitter, serial versus overlapped rollouts, observation/reward/
reset/recurrent-state alignment, CUDA graphs versus eager across three pool
versions). `sampled_check` passed on the new build (459 decisions, three episodes
ending inside the first phase) and on the old build through the transparent
fallback; `check_telemetry.py` matched MAIN's build exactly for state, observations
and replays on both the sampled and split paths; the new synchronous loop matched
the previous loop bit-for-bit over 24 updates.

Reset cost (`reset_profile`, `frame_profile`): a reset is about 32 ms, of which
scene load is 13 ms and the 240 warmup ticks are 19 ms at 16 ticks per engine
frame. Engine frame overhead is only about 0.6 ms, so lifting the warmup
ticks-per-frame could save at most 9 ms per reset. The final runs completed 486
matches, so the bound is 4.4 s per 393k-decision run (6% of the 71 s active time
of the default arm) if no reset stall is hidden by the other group's inference;
overlap was not instrumented, so that is an upper bound, not a measurement.
Reset semantics were left unchanged; this remains a candidate for later work
since it needs a game rebuild and a parity re-check.
