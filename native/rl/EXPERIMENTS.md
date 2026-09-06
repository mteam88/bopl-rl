# Dash / Arrow / Platform experiments

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
