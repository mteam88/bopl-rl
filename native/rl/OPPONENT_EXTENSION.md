# Bounded A/B continuation

This study tests whether the script mixture's advantage survives direct competition
and longer training. It attempts to continue all three A and B seeds from
[the one-million-decision experiment](evidence/opponent-league-001/REPORT.md) to
four million learner decisions each. A retains current and historical opponents;
B retains the same mixture with scripts. There is no Bow-initialized C arm here.

The [live dashboard](https://powerspec-g728.tailb55536.ts.net:8876/) shows nested
study progress, every available checkpoint evaluation, direct crossplay, saved
game clips, and the fixed failure review. Its study selector retains the earlier
A/B/C comparison; **Earlier runs** opens the original replay dashboard. The match
browser plays full saved native replays even without a rendered video, with
filters, seeking, frame stepping, and speed controls. Replay preparation runs the
recorded inputs in the native game and verifies the replay hash and final state.
Only games with recorded inputs are available, as the coverage count explains.

## Fixed comparison

The [protocol](evidence/opponent-extension-001/protocol.json) identifies source
checkpoints, source code, game binaries, budgets, and evaluation schedules. It was
written before the continuation began. The six runs retain seeds 73, 173, and 273,
eight game workers, 128-decision rollouts, 32-decision recurrent sequences, three
PPO epochs, and learning rate 0.0003. Only the training budget increases.

Each source has 1,001,472 learner decisions. The planned finish is 4,001,280
because a rollout contains 1,536 learner decisions. This would add 2,999,808
decisions per run and 17,998,848 across six runs. Opponent decisions remain excluded from
PPO and the learner budget. Dash, Bow, Platform, jumping, the full held-button
space, and joystick aiming remain available at 15 Hz with a nine-tick delayed
observation. Training maps remain 0, 3, and 6.

Continuation restores policy weights, Adam state, Torch CPU/CUDA random state,
step/update counts, the latest four historical checkpoints, and their snapshot
cadence. Games restart, recurrent memories start at zero, and the private NumPy
random generator restarts from the run seed. This is a declared continuation
boundary, not an exact resumption of interrupted game trajectories. Both arms
use the same boundary. The initial saved continuation checkpoint is checked
bit-for-bit against the source for all restored state.

## Interrupted trial and common checkpoint

B seed 73 aborted during collection after 3,841,536 completed learner decisions:
99 entities exceeded the fixed observation capacity of 96. Its last saved policy
has 3,838,464 learner decisions. No 4M checkpoint exists. The failing state was
not truncated, reset, or retried, and the model and frozen settings stayed fixed.
The other five trials continued to the original budget.

The [capacity fallback protocol](evidence/opponent-extension-001/capacity-fallback-protocol.json)
was declared before the supplemental evaluations. It selects the highest
originally planned checkpoint available for all six runs, using availability
rather than scores. That checkpoint is 3M (3,001,344 actual learner decisions). All nine direct seed pairings and all
six reserved-map script suites are repeated there: 15 supplemental suites and
3,168 games. This produces a balanced 1M-to-3M comparison. Available 4M results
remain visible per seed; they cannot provide the intended full three-seed A/B
comparison. Five original 4M suites, totaling 816 games, are unavailable.

A separate interruption affected A seed 173 when an operator closed its failed
supervisor session, terminating the still-running child process group. This was
not a game-capacity failure. Its attempt was preserved, then repeated from the
original 1M checkpoint with identical settings. The original and replacement
2M and 3M states were compared for exact policy, optimizer, Torch random state,
and counters. Reports tied to the original attempt remain archived; replacement
reports use the replacement checkpoint hashes. This repeat was not selected by
performance and is recorded separately from the six experimental trials.

## Evaluation

Development matches are measured at 1M, 2M, 3M, and 4M: three maps, three fixed
scripts, eight action-seed pairs, both sides, or 144 matches per run/checkpoint.
Reserved-map matches are measured at 1M and 4M: three maps, four fixed scripts,
16 pairs, both sides, or 384 matches per run/checkpoint. All existing 1M script
results are reused. These benchmarks have already been inspected; they now
track progress and are not a new blind test. No checkpoint is selected by score.

Direct crossplay at 1M and 4M covers all nine A/B training-seed combinations,
six maps, eight action-seed pairs, and both player sides: 864 matches at each
budget. The seeds and sides are paired across budgets. The evaluator records A
in its learner field and B in its opponent field; neither is updated during
these matches. The original extension schedules 42 new suites and 6,624 new
matches, alongside the reused script evidence. The capacity fallback raises the
scheduled total to 57 suites and 9,792 games; 52 suites and 8,976 games are
available after excluding the five unavailable 4M suites. Archived evaluations
of the operator-interrupted attempt are additional repeated work, not additional
independent benchmark games.

Wins use all matches as their denominator. Timeouts and simultaneous shared
winner flags count as draws. Raw game flags remain preserved. The game's existing
training reward for simultaneous winner flags remains unchanged in both arms.

Plots show each training seed and condition means. Descriptive bootstrap
intervals resample paired training seeds and within-cell action-seed blocks,
keeping both player sides together. Crossplay resamples the A and B seed axes
separately because pairings share policies. Maps and scripts stay fixed. Three
seeds per condition provide limited evidence about broader generalization.

## Fixed failure review

The initial protocol selected the first B loss, sorted by map/action seed/side,
for each training seed against archers and evasive scripts, plus its matched A
case regardless of outcome. All 12 initial cases fell on map 1. Before analyzing
traces, a [supplemental protocol](evidence/opponent-extension-001/failure-review-protocol.json)
added the same selection within maps 2 and 5. The resulting 36 cases contain
18 selected B losses and 18 matched A cases.

Each original checkpoint and action seed is replayed with unchanged observations.
Analysis records full state and actions separately, then verifies final state,
both players' telemetry against the original game. All 36 cases match those
measurements. The 26 cases with an originally saved replay also match its hash;
the other ten receive a new replay recording without a prior byte comparison.
The analysis does not feed those extra state fields into either controller.

[Failure measurements](evidence/opponent-extension-001/failure-overview.png)
cover shot/hit counts, death attribution, contiguous Bow input holds, observed
projectile speeds, and activation-to-shot timing. Input holds include time outside
active Bow. Timing has four-tick resolution. First-observed projectile velocity
is not exact muzzle velocity. Initial heading relative to the current opponent
center does not account fully for leading, gravity, or intervening geometry.
This selected sample diagnoses failure modes; it is not an unbiased strength
estimate or a causal experiment.

The original [4M follow-up plan](evidence/opponent-extension-001/failure-review-4m-protocol.json)
was preserved but could not run for all six policies. A
[3M follow-up](evidence/opponent-extension-001/failure-review-3m-protocol.json)
repeats exactly these 36 case names at the common checkpoint, regardless of
their new outcomes. It
records additional traces of already scheduled matches; it does not add primary
benchmark games or reselect failures. This checks whether the 1M diagnostic
observations still describe the longer-trained policies.

The game code connects these measurements: `BowTransform.Shoot` multiplies arrow
speed by `loadingFrame + 1`, with loading frame ranging from zero to four.
`OldUpdate` latches button release and then stops updating the aim vector, even
when the actual shot must wait for its animation threshold. Short holds can
therefore produce weak shots with an early-frozen aim. This supports a timing
hypothesis; a controlled intervention is still needed to measure its effect.

## Reproduce

Use a fresh output directory, the source one-million-decision artifacts, and the
existing research environment. The preparer writes a new protocol rather than
overwriting this study. It also derives the recent opponent pool from source
checkpoint metadata.

```bash
native/artifacts/rl/venv/bin/python native/rl/research/prepare_extension.py \
  --output native/artifacts/rl/opponent-extension-new
native/artifacts/rl/venv/bin/python native/rl/research/run_extension.py \
  --output native/artifacts/rl/opponent-extension-new
native/artifacts/rl/venv/bin/python native/rl/research/evaluate_extension.py \
  --output native/artifacts/rl/opponent-extension-new --jobs 2 --workers 2

```

The unchanged primary run can reproduce the observation-capacity failure. An
incomplete directory is intentionally not overwritten. Inspect a failure before
using the availability queue or continuing untouched trials; do not silently
retry a data-dependent abort. The operator-retry reconciliation helper is specific
to the recorded A seed 173 incident and is not a general resumption command.

To recheck and regenerate analysis from the preserved completed study:

```bash
native/artifacts/rl/venv/bin/python native/rl/research/audit_extension.py \
  --output native/artifacts/rl/opponent-extension --include-final-review
python3 native/rl/research/plot_extension.py \
  --artifacts native/artifacts/rl/opponent-extension \
  --output native/rl/evidence/opponent-extension-001
python3 native/rl/research/analyze_league_failures.py \
  --artifacts native/artifacts/rl/opponent-extension \
  --review-name failure-review-3m --matched-later \
  --output native/rl/evidence/opponent-extension-001/matched-3m
```

Training and evaluation may run in separate processes. On this host, two
training processes were overlapped after checking memory and measured aggregate
throughput. The game/model settings stayed fixed. Per-run wall durations overlap
and must not be summed as the elapsed duration of the study.

The evaluation queue waits
for atomic checkpoint files. Completed suites are checked and reused; incomplete
outputs require inspection instead of being overwritten. Raw absolute paths
retain the original workspace provenance; analysis resolves checkpoints from
the artifact location after preservation.

To show a study while it is running in an isolated workspace:

```bash
python3 native/rl/research/serve.py --host 127.0.0.1 --port 8876 \
  --study-root opponent-extension=/absolute/workspace/native/artifacts/rl/opponent-extension
```

The dashboard reads training and evaluation artifacts. Preparing a replay writes
only its derived cache and conversion logs, with one preparation worker at a time.
The live service can return to its default artifact root after completed results
are copied and checked in the working checkout.
