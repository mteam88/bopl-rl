# Native SET trainer handoff

The trainer uses native PufferLib recurrent PPO, the Bopl entity/slime set
encoder, native minGRU, and match-stable frozen opponents. The 400M run is owned
by BB thread `thr_wq22hezkbf`; the command below is its launch configuration,
not a record that the run has started.

## Launch configuration

Run from the qualified Unity checkout (the directory containing `Assets/`).
The native CUDA binding, CPU policy evaluator and IL2CPP player must be built
there as described in [README.md](README.md). A copied virtualenv must have its
editable PufferLib install relinked with `puffer/setup.py`; do not accidentally
import a binding from another checkout.

```sh
native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
  --output native/artifacts/rl/null-hypothesis-native/001-s73 \
  --game Builds/LinuxIL2CPP/BoplBattle.x86_64 \
  --encoder set --hidden 128 --layers 1 \
  --steps 400000000 --seed 73 \
  --workers 16 --buffers 2 --banks 4 \
  --horizon 128 --learner-passes 1 --learning-rate 0.0003 \
  --training-maps 0,1,2 --evaluation-maps 0,1,2 --heldout-maps 3,4 \
  --episode-ticks 7200 --entropy 0.01 --entropy-floor 0.0005 \
  --entropy-half-life 30000000 \
  --checkpoint-every 1000000 --eval-every 5000000 \
  --evaluation-games 8 --heldout-games 4 --frozen-games 16 \
  --evaluation-workers 2
```

Rewards are fixed by this entrypoint: +1 for winning, -1 for losing, -0.5 for
both players on timeout, and zero kill/death shaping. Half the games use live
self-play, and half pair one learner with a frozen opponent. Frozen banks
change only after their active matches end.

Entropy uses `max(0.0005, 0.01 * 2**(-N / 30000000))`, where `N` counts learner
decisions completed before the update. It reaches 0.005 at 30M, 0.0025 at 60M,
and the floor at about 129.7M. The schedule is independent of `--steps`. The
native loss reads its updated device scalar during CUDA graph replay; an
accelerated six-update check verified every halving and the floor. Learning
rate retains native cosine decay from 0.0003 over the complete run budget;
`anneal_lr=1` and `min_lr_ratio=0` are explicitly recorded in the native config.

Budgets and checkpoint/evaluation cadence count learner decisions. With this
configuration an update contains 3,072 learner decisions; completing the last
rollout gives 400,002,048 learner decisions in 130,209 updates. Metrics `step`
and checkpoint filenames count physical player decisions, including frozen
players: 533,336,064 at that endpoint. Both counters are saved explicitly.

Evaluations are synchronous between updates and add wall time. The full suite
above contains 56 scored games (48 on training maps and eight on held-out maps)
plus one self-play replay. It recorded 18.7 seconds on an early SET checkpoint
with two evaluation workers, plus process startup; stronger policies and
longer matches can change that cost. Evaluation and checkpoint cadence are
independent: this command schedules about 80 evaluations and 400 checkpoints.
The final checkpoint is evaluated even when it falls between intervals. The initial
native policy is saved automatically as the baseline. Optional frozen
evaluation opponents require a schema-2 manifest of compatible native
checkpoints. Files are immutable; existing output directories are rejected.

## Qualification

The cleaned IL2CPP game passes a 12-game deterministic comparison against the
previous compiled player: exact state, delayed observation and replay bytes.
Sparse reward, a forced timeout, event conservation and reset checks pass.
A normal rendered replay was also inspected. Runtime identities and evidence
paths are in [native-cutover-game.json](evidence/native-cutover-game.json).

Final native checks on that player pass: sampled-step correctness (459
decisions and three episodes ending in the first sample phase), reset isolation
(5,738 snapshots), linear/SET/16-worker SET smokes, bank routing, and unit tests.
The SET test covers forward/backward against an independent oracle, masking,
permutations, empty pools and CUDA graph reuse. Central finite differences of
the actual native forward also match the native backward at 64 coordinates
across all four weight matrices; maximum absolute error is below 0.000076.

On the final game, three interleaved runs per encoder measured 14,004 learner
decisions/second including startup for linear and 13,683 for SET (medians).
SET costs 2.3% throughput in this comparison. Each run completed 786,432
learner decisions; none hit the cgroup memory high limit. These measurements
exclude diagnostic evaluation work.

The canonical working checkout also passes a fresh CUDA/CPU build and a real
16-worker SET training/evaluation smoke: 12,288 learner decisions, 17 matches,
five hash-validated checkpoints, nine replays and nine policy-value dumps.
The live dashboard runs from that checkout and passed its browser checks.

Matthew selected SET and removed the 60M matched comparison as a launch gate.
No such run was launched. The corrected exponential schedule also passed a
fresh canonical training/evaluation smoke; evidence is in
[native-cutover-entropy.json](evidence/native-cutover-entropy.json).

## Limits and follow-up

This is a Dash/Bow/Platform diagnostic, not evidence of whole-game competence.
The previous trainer's timing uses a different recurrent model, optimizer,
reward setup and initialization, so it does not isolate backend speedup.
Long-run stability and learning at 400M decisions are the experiment's purpose.
The optional `autoresearch/strength_check.py` comparison rejects incompatible
cached plans, settings and checkpoint identities. It can be used for a later
A/B; its short-budget learning-rate schedule is not a prefix of the 400M run.

After launch, implement native archer/pursuer/evasive evaluation opponents.
These provide additional fixed anchors once random opponents stop being
informative. They are a follow-up, not a prerequisite for this run.
