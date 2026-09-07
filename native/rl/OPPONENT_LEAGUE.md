# Scripted-opponent comparison

This experiment tests whether algorithmic opponents improve full-match learning,
and whether the existing Bow teaching example adds value. The long-term target
remains superhuman play across Bopl Battle, including discovering and countering
new combinations. This first comparison covers two players with dash, Bow and
platform on six maps. It does not establish broader ability compatibility or
adaptive league training.

[Completed results, plots and clips](evidence/opponent-league-001/REPORT.md).
[Frozen protocol](evidence/opponent-league-001/protocol.json).

## Conditions

| Condition | Initialization | Opponent assignments |
|---|---|---|
| A | Random network | Existing current/current and current/historical mixture |
| B | Random network | Current/current, current/script and current/historical |
| C | Bow teacher network | Same mixture as B |

Each condition uses seeds 73, 173 and 273. Runs have a target of one million
**learner decisions**, counting only actions included in PPO. Complete rollouts
can overshoot that target slightly; both learner and total-player counts are
recorded. All use the existing recurrent PPO, GRU128, 3,257 observations, 15 Hz
controls and nine-tick world delay, with full DAP controls available.

The full-game reward remains +1/-1 for a win/loss, +0.1 for a kill and -0.1 for a
death. Timeout has no winner bonus and zero terminal bootstrap. Actual wins,
losses and timeouts are reported separately from training reward. All three
conditions keep learning rate 3e-4; this is a test of the existing full-game
optimizer, including its potential to forget the initialization.

Eight workers run three-stock matches with a 3,600-tick limit. Half the workers
play current/current. In B/C, each of the remaining workers chooses a script
with probability 2/3 or a historical policy with probability 1/3 at match reset.
Opponent side is reassigned at reset. An assignment lasts the complete match.
Actual game-decision exposure is logged because match duration affects the
realized mixture. Historical snapshots are retained while a match uses them.

## Script interface and styles

`ScriptedOpponent.act(observation)` consumes only one player's delayed feature
vector and its private state/RNG. `Controller` provides the same single-player
evaluation interface for scripts and neural policies. The trainer retains
batched GPU inference for historical policies and adapts scripts to the same
observation/action rows. All fixed-opponent rows are excluded from PPO.

- **Archer:** charge aimed shots and move toward or away from a preferred range.
- **Pursuer:** approach before charging, with occasional dash and jump inputs.
- **Evasive:** prefer more separation and retreat/dodge when a visible projectile
  is on an approaching trajectory; shoot between evasions.

Each has a second, more patient training setting. Shot error and dash choices
use a separate seeded RNG. Reaction interval, range, charge duration, accuracy
and aggression are fixed in the manifest. Scripts use approximate projectile
prediction, not current simulator state or future trajectories. Their movement
and recovery rules are deliberately simple and remain exploitable.

The reserved **switcher** chooses pursuit or evasion based on visible projectile
threats. It is an unseen behavior combination, not an unseen ability or a novel
strategy invented by the learner.

## Bow initialization

The source is `bow-curriculum/teacher-fit/final.pt`: a network fitted to a
38-decision scripted shot for 2,000 supervised updates. No curriculum PPO
checkpoint is used. `initialize_league.py` expands its two relevant button
logits to all 16 combinations while preserving the Bow probability exactly.
Jump, dash and platform each start with independent probability 0.001. All
classes have nonzero support and all output weights remain trainable.

This is a bundled initialization intervention: it changes the representation,
aim/Bow outputs and initial preference for other controls. C versus B cannot
isolate which of those components helped. All runs start a fresh Adam optimizer.
The source fit cost is reported separately from the new game-experience budget.

The expanded network passed the original shot with all controls available in
32/32 stochastic trials and an argmax trial under full-game observations. That
is a fixed-setup initialization check, not generalization evidence.

## Frozen evaluation

Training and development use maps 0, 3 and 6. Final evaluation uses maps 1, 2
and 5, with no training on them. Development uses three script settings excluded
from the training manifest; final evaluation uses another three settings plus
the switcher. Side-swapped pairs share reset seeds and use explicit action RNG
seeds. Different reset seeds on one map do not create independent layouts.

For every final checkpoint:

- Development: 8 seed pairs × 3 bots × 3 maps = 144 matches.
- Final: 16 seed pairs × 4 bots × 3 maps = 384 matches.
- Retention: 32 stochastic trials and one argmax trial of the original shot,
  with unrestricted controls and the full-game observation clock.

All nine final checkpoints are evaluated. The final set is not used for tuning
or checkpoint selection. Report all seeds and per-opponent/per-map results,
including losses, timeouts and worst-opponent performance. Paired comparisons
are B minus A and C minus B. Three training seeds limit statistical confidence.
Saved native replays allow ordinary game playback.

## Reproduction

Run from the exported project, using `native/artifacts/rl/venv/bin/python`.
The existing game assembly and native libraries must match the recorded hashes.

Use a new output directory; the commands below use `opponent-league-rerun`.

```bash
mkdir -p native/artifacts/rl/opponent-league-rerun
cp native/rl/evidence/opponent-league-001/qualified-bots.json \
  native/artifacts/rl/opponent-league-rerun/train-bots.json

python native/rl/research/initialize_league.py \
  --source native/artifacts/rl/bow-curriculum/teacher-fit/final.pt \
  --output native/artifacts/rl/opponent-league-rerun/bow-initial.pt

# prepare_league.py consumes the qualified three-bot train-bots.json.
python native/rl/research/prepare_league.py \
  --output native/artifacts/rl/opponent-league-rerun

python native/rl/research/run_league.py \
  --output native/artifacts/rl/opponent-league-rerun --phase train

python native/rl/research/parallel_league_evaluate.py \
  --output native/artifacts/rl/opponent-league-rerun --jobs 3 --workers 2

python native/rl/research/league_retention.py \
  --output native/artifacts/rl/opponent-league-rerun
```

Output directories are never overwritten. The series runner can skip complete
runs but stops on an incomplete run for inspection. Source hashes, protocol,
checkpoint identities, configs, logs and replays remain with the artifacts.
Recorded absolute paths identify the original run; series tools resolve saved
checkpoints from the local artifact directory after a move.
The evaluator's low-level script/policy schedule can also be used independently
through `league_evaluate.py`.
