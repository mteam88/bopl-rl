# Human replay imitation

The first experiment can fit a human demonstration with the existing GRU and
execute the learned policy in the real game. It does not yet transfer well to
other demonstrations. Resampling the human's controls is itself a substantial
source of changed behavior, before any model is involved.

[Measured results and figure](evidence/human-imitation-001/REPORT.md).

## Data contract

`research/replay_data.py` decodes checksummed BOPLREP2 files and reconstructs the
recorded setup and every 60 Hz input through the public C game API. It validates
source/catalog pins against the running game; the dataset also records the
actual game assembly and C encoder hashes. The original player order, input mode,
colors, teams, map, stocks and ability slots are preserved. The current extractor
rejects keyboard targets, unresolved abilities and Random loadouts. It never
silently identifies a human from controller type: select `--player` explicitly.

The shared C observation encoder provides all 3,257 features. Extraction begins
at tick 240, samples a target every four ticks, and uses world state from nine
ticks earlier, clamped to tick 240 during startup. The episode-clock feature
subtracts those 240 ticks and uses the training limit of 3,600. The immediate
previous-action fields contain the previous sampled decision, never the current
label. Original inputs between decision boundaries still drive reconstruction;
`action_resampling_mismatch` reports how often repeating the sampled action would
change a source tick. This is a supervised dataset of original human states,
not a claim that quantized actions recreate those states.

Each NPZ contains observations, actions, original tick indices, alive masks,
arrow-event counters and source metadata. Whole replays form the train/validation
split. Dead frames have zero loss weight while the GRU still processes the
chronological sequence. The initial experiment uses full-episode supervised
backpropagation, Adam at 0.001 and no entropy bonus. The actor/critic architecture
is unchanged; the critic is not trained by this objective. Imitation checkpoints
can be loaded for inference, but are **not PPO resumptions** and cannot be passed
to `train.py --load` (which expects optimizer state).

## Reproduce

Run from the exported Unity project. Use the existing RL virtualenv. The game
build must already support the RL bridge and share the replay's simulation pin.

```bash
make -C native/rl research

native/artifacts/rl/venv/bin/python native/rl/research/check_demonstrations.py \
  --output native/artifacts/rl/demo-contract

native/artifacts/rl/venv/bin/python native/rl/research/verify_demonstration.py \
  /absolute/path/498.rep --output native/artifacts/rl/demo-parity

native/artifacts/rl/venv/bin/python native/rl/research/extract_demonstrations.py \
  /absolute/path/498.rep /absolute/path/499.rep \
  --player 2 --output native/artifacts/rl/demo-data

native/artifacts/rl/venv/bin/python native/rl/research/imitate.py \
  --train native/artifacts/rl/demo-data/498-p2.npz \
  --validation native/artifacts/rl/demo-data/499-p2.npz \
  --output native/artifacts/rl/demo-fit --epochs 2000

native/artifacts/rl/venv/bin/python native/rl/research/evaluate_imitation.py \
  /absolute/path/498.rep /absolute/path/499.rep --player 2 \
  --checkpoint fit=native/artifacts/rl/demo-fit/final.pt \
  --output native/artifacts/rl/demo-evaluation

native/artifacts/rl/venv/bin/python native/rl/research/probe_demonstration_cadence.py \
  /absolute/path/498.rep /absolute/path/499.rep --player 2 \
  --output native/artifacts/rl/demo-cadence
```

Evaluation replays the original prefix through tick 240, then gives the model
control of the selected player using its own recurrent memory and previous
actions. Other players retain their recorded inputs. Each variant gets the same
initial setup and original replay tick budget, or stops at an earlier terminal.
`human60` and `human15` are input-playback controls. Policy evaluation uses argmax.
This is a counterfactual replay diagnostic against fixed opponent input, not a
competitive strength benchmark. Exported `.rep` files can be rendered with the
existing `research/render_video.py`.

`check_demonstrations.py` compares extraction with live evaluator observations
bit-for-bit, including startup delay, previous input, and episode-clock scaling.
`verify_demonstration.py` separately compares full managed gameplay traces with
ordinary replay playback, using the existing audited presentation exclusions.
The former is an observation-contract test; the latter checks reconstruction.
