# Dash / Arrow / Platform experiments

## Throughput — 2026-09-06

Runtime changes before the multi-map series: one game round trip per decision,
two pipelined rollout groups, and CUDA-graph inference. Matched 393,216-decision
runs at the series contract (16 workers, horizon 128, 32 minibatch segments,
three snapshots) measured 5,506 active decisions/s against 4,448 for the previous
synchronous loop (+24%; +15% wall including startup), with the serial oracle of
the same configuration bit-exact across all 96 updates in all three repeats.
Details and limits in [README](README.md#evidence-and-limits) and
[evidence/throughput/summary.json](evidence/throughput/summary.json).

## dap-series-001 — 2026-09-06

Three independent 10M-decision runs completed. All three final policies beat
both weak fixed opponents on the training map. The checkpoint diagnostics show
most of the gain between 1M and 3M decisions; improvement from 3M to 10M is mixed.
Transfer to the second untrained map remains weak.

[Dashboard, behavior graphs and videos](https://powerspec-g728.tailb55536.ts.net:8876/?series=dap-series-001)
(Tailscale required). [Comparison figure](evidence/dap-series-001/overview.png),
[SVG](evidence/dap-series-001/overview.svg),
[machine-readable summary](evidence/dap-series-001/summary.json).

### Fixed protocol and completion

The series used revision `4d11f1e`, seeds 73, 173 and 273, and the same PPO,
reward, observation, opponent-checkpoint cadence and 16-worker settings as the
pilot. Telemetry was added without entering policy observations or rewards.
All runs shared the frozen pilot starting-policy opponent. Source, assembly,
bridge and baseline hashes were checked across all three configurations.

- Each run completed 10,002,432 agent decisions and 2,442 updates.
- Total: 30,007,296 agent decisions, 22,505,472 learner decisions and 28,156 matches.
  Historical-opponent decisions are excluded from learner counts and PPO losses.
- Collection plus PPO averaged 4,132 / 4,115 / 4,127 decisions/sec by seed.
  Whole-series elapsed time, including evaluations and media, was 8,076 seconds.
- All training metrics were finite, with zero game errors or observation overflow.
- Each seed has four 161-game diagnostic suites and a fresh 641-game final suite.
  Suites include one self-play replay, which is not a strength measurement.
- Saved media: 12 native game videos and 75 native replays, with physics replays.

### Fresh final evaluation

Each trained-map opponent used 256 games (128 side-swapped seed pairs). Each
untrained map used 64 games (32 pairs), against random play. These game seeds
were separate from checkpoint diagnostics and shared across training seeds.
Timeouts count as non-wins. The figure uses 95% bootstrap intervals that resample
whole side-swapped pairs, separately for each fixed trained policy.

| Training seed | Random, trained map | Starting policy, trained map | Random, untrained map 1 | Random, untrained map 2 |
|---|---:|---:|---:|---:|
| 73 | 208/256 (81.3%) | 216/256 (84.4%) | 44/64 (68.8%) | 34/64 (53.1%) |
| 173 | 190/256 (74.2%) | 196/256 (76.6%) | 41/64 (64.1%) | 37/64 (57.8%) |
| 273 | 210/256 (82.0%) | 211/256 (82.4%) | 48/64 (75.0%) | 27/64 (42.2%) |

Timeout counts in the four table columns were respectively 5/5/2/0, 6/4/4/0,
and 6/6/4/0. Do not pool these training seeds into an independent-game estimate
or interpret beating random and untrained policies as human-level ability.

The environment-death curves drop from roughly 3.7 to 1.7 deaths per simulated
learner-minute. Fixed-opponent diagnostics also show fewer environment deaths
at 3M than at 1M in every run. This supports improved survival as an important
part of the gain; it does not isolate the contribution of aiming or tactics.
The longer budget answered the pilot's immediate question: learning becomes
visible after 1M. It does not establish that simply training beyond 10M will
resolve transfer or improve play against stronger opponents.

### Profiling and validation

The requested Claude Code Fable 5.1 subagent profiled collection and removed
repeated historical-opponent indexing work and unnecessary categorical
validation synchronizations. In three matched 65,536-decision runs per arm,
median collection-plus-PPO throughput increased from 4,319 to 4,773 decisions/sec
(**10.5%**). These short comparisons used the pre-telemetry bridge and excluded
evaluation and video; the actual series rates above include the new telemetry.
[Performance evidence and limitations](evidence/collection-performance.json).

The subagent checked identical per-update metrics, final weights and CPU/CUDA
random-generator state across all six comparison runs. Opponent-cache equivalence
passed 40 trials on CPU and 40 on CUDA. Integration checks compared telemetry
conservation, delayed observations and lossless replays against the prior build;
the controlled-series smoke exercised training, final evaluations and videos.
Desktop and phone dashboard checks verified video playback, advancing physics
replays, behavior charts and no JavaScript errors or horizontal overflow.

This experiment runs the reconstructed Unity game through the native C bridge;
it does not complete the separate Rust gameplay port.

## dap-001 — 2026-09-06

The first GPU research run completed successfully, but has not established a
reliable playing-strength improvement. The training and observation pipeline is
working; the policy remains close to its untrained opponents in this evaluation.

[Dashboard and replays](https://powerspec-g728.tailb55536.ts.net:8876/?run=dap-001)
(Tailscale required). [Machine-readable results](evidence/dap-001.json).

The run used revision `6534da1`, 16 persistent game workers, recurrent PPO on CUDA,
two players, Dash + Arrow + Platform, 15 Hz decisions, and a 150 ms delayed world
view. Loadouts and cooldowns were public. Half the games used current self-play;
half played recent historical checkpoints. See [the protocol](README.md).

- 1,003,520 agent decisions, 245 updates, 1,215 completed matches.
- Collection and training took 311 seconds, averaging 3,223 decisions/sec.
- Total process time, including startup, evaluations and final video, was 408 seconds.
- All logged metrics were finite; zero reported game errors or observation overflow.
- The training service exited with code 0. The read-only dashboard remains running.
- Eleven checkpoint evaluations have downloadable native and physics replays;
  nine also have native game videos, including the initial and final policies.
- Browser checks passed on desktop and phone sizes, including actual native video
  playback, physics replay advancement, and no JavaScript errors or horizontal overflow.

The isolated worker comparison measured 3,252 versus 4,490 median decisions/sec
for eight versus sixteen workers, a 38% increase. The actual research run includes
CPU contention from evaluation and native video rendering.

### Expanded final evaluation

After the run, the frozen final checkpoint played 64 games against each baseline
on the training map, using 32 seeds with player sides swapped. These include the
four diagnostic seed pairs; this is an expanded check, not an independently
held-out tournament. No policy weights changed during evaluation.

| Opponent | Wins | Win rate | Descriptive 95% Wilson interval | Timeouts |
|---|---:|---:|---:|---:|
| Random buttons | 32 / 64 | 50.0% | 38.1–61.9% | 1 |
| Initial policy | 36 / 64 | 56.3% | 44.1–67.7% | 1 |

Timeouts count as non-wins. The intervals do not adjust for correlation within
side-swapped seed pairs. On each unseen map, the final checkpoint won 2/4 games
against random play; these samples are too small to assess generalization.

The small checkpoint diagnostics varied substantially. The final diagnostic
reported 6/8 versus random and 5/8 versus initial, which looked more promising
than the expanded result. Do not select a best checkpoint from that noisy curve
and call it a strength gain.

This pilot establishes a reproducible baseline. The next strength experiment
should predeclare a longer budget, use multiple training seeds, and evaluate
frozen final checkpoints on fresh seed pairs. Keep this run as the reference;
do not silently change its reward or observation contract. This run uses the
reconstructed Unity game through the C bridge; it does not complete the separate
Rust gameplay port.
