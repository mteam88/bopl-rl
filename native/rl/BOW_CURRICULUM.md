# Bow curriculum within the full-game research goal

[First completed experiment: results and video](evidence/bow-curriculum-001/REPORT.md).
[Full-game opponent comparison and retention test](evidence/opponent-league-001/REPORT.md).

The long-term goal is a bot capable of superhuman play across all of Bopl Battle,
including discovering new combinations and strategies and adapting to defend
against them. A fixed opponent, one map, or one successful ability is only a
learning diagnostic. Human demonstrations can help exploration without defining
the ceiling of the final policy.

The first experiment used a small task in the reconstructed Unity game: hit an
idle opponent with Bow. The opponent still obeys normal gravity and platform
motion. Training uses the same 3,257 delayed observations, GRU128 and recurrent
PPO update as the full-game experiment. The raw controls in this stage are the
256 stick encodings and Bow held/released; jump, dash and platform are temporarily
excluded. No combat macro is installed into the normal policy.

## Stages and evidence gates

1. Qualify a repeatable starting setup with a scripted shot. A miss by an
   untrained policy is informative only if the intended shot is feasible.
2. Learn aim and the Bow button under autonomous game control. Reward an actual
   accepted arrow hit, not firing or matching a recorded button sequence.
3. If that fails, temporarily prescribe charge/release timing to isolate aiming.
   Exclude assisted buttons from the policy loss. Remove that assistance before
   claiming learned control of a complete shot.
4. If timing remains difficult, initialize from a short, verified teacher
   trajectory at the exact control rate. Evaluate with the teacher removed;
   distinguish imitation initialization from skills discovered by PPO.
5. Once one setup is reliable, vary starting positions and target motion, then
   maps. Evaluate on setups excluded from fitting, with separate action RNG seeds.
6. Reintroduce movement and other abilities, then test whether the learned skill
   improves full-game results against a fixed diverse opponent set.

A fixed-layout pass is at least 90% successful hits across 128 stochastic trials
and a successful argmax trial. Repeated seeds on maps with identical initial
positions measure sampling reliability; they are not independent layouts or
proof of generalization. Stage assistance is recorded in every checkpoint and
trial result.

## Task contract

A trial ends on the first accepted learner arrow hit, either player's death, or
its time limit. Reward is +1 for a hit, otherwise -0.25 for a learner death and
zero for all other outcomes. A target falling by itself is not success. Every
terminal uses zero bootstrap value. The full-game C rewards remain unchanged.

Trials use ordinary game resets and the research C encoder, with four-tick
controls and nine-tick world delay. The learner receives its own actual previous
action. The optional timing script uses only that delayed observation and a
counter. The full-game trainer and this runner import one shared PPO update;
its default result was compared bit-for-bit with the prior implementation on
CPU and CUDA before curriculum training.

The saved native replays can be played in the original game. Checkpoint metadata
includes task, action restrictions, assistance, source hashes and assembly hash.
Bow checkpoints need the Bow evaluator/action distribution; they are not directly
comparable to full-game checkpoints and are not full-game optimizer resumptions.

## Longer-term progression

The eventual evaluation must cover all supported abilities and their combinations,
maps, player counts and opponent styles, with explicit compatibility checks.
Persistent archives of distinct opponents and deliberately trained counterstrategies
should test whether improvements survive opponents that exploit them. Averages
against random play or a few recent copies of the learner cannot establish this.

At each expansion, measure retention of older skills as well as the new capability.
Reserve combinations and strategies for transfer tests. Claims of superhuman play
will require human competition under a declared observation/control contract;
structured simulator state is more precise than human vision, even with a delay.

This curriculum does not by itself implement that opponent population, all-ability
qualification, or broad adaptation. It creates a small, falsifiable learning test
before committing more compute to those stages.

## Commands

Run from the exported Unity project, using `native/artifacts/rl/venv/bin/python`.

```bash
native/artifacts/rl/venv/bin/python native/rl/research/probe_bow_task.py \
  --output native/artifacts/rl/bow-probe --maps 0 --seeds 101,102 --offsets 4

native/artifacts/rl/venv/bin/python native/rl/research/train_bow.py \
  --output native/artifacts/rl/bow-raw --steps 262144 --workers 8

native/artifacts/rl/venv/bin/python native/rl/research/train_bow.py \
  --output native/artifacts/rl/bow-aim --steps 262144 --workers 8 --assistance timing

native/artifacts/rl/venv/bin/python native/rl/research/evaluate_bow.py \
  --checkpoint /absolute/path/policy.pt --output native/artifacts/rl/bow-eval \
  --trials 128 --start-move-ticks 16
```

A signed `--start-move-ticks` evaluation prefix moves the learner toward or away
from the opponent using ordinary game inputs before control begins. It changes
actual positions and cooldown timing, preserves the observation history, and
adds no information to the policy. It is a combined setup variation, not a pure
position-only intervention. Source and destination coordinates are recorded.

`--hold-prior .95` changes only the initial Bow-head bias, making holding likely;
it never sets an action duration. `--load` starts from weights with a fresh Adam
optimizer; `--load-optimizer` also retains Adam state, but still starts fresh game
trials and recurrent memories. Each output directory must be new.

The first successful seeded runs use `--load <teacher-fit/final.pt> --lr .00003`
with `--assistance none`. A baseline run at .0003 lost the skill in one of two
seeds. Keep this learning-rate result scoped to the drill; it does not change the
full-game trainer's defaults. The runner also accepts `--opponent approach` and
`--start-move-ticks` for subsequent curriculum tasks.
