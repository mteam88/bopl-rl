# Dashboard metrics contract

What the research dashboard (`serve.py`, `study_data.py`, `web/`) reads from a
training run directory. The native trainer must keep writing everything marked
**required**; fields marked **new** are additions the dashboard renders as soon
as they appear. Owned by the dashboard side; change it here first, then in the
trainer.

## Run directory layout (required)

```
<run>/config.json           written once at start (atomic)
<run>/status.json           rewritten every update (atomic)
<run>/metrics.jsonl         one JSON object per optimizer update, append-only;
                            a torn final line is tolerated, torn middle lines are not
<run>/checkpoints/policy-%09d.pt        (step-stamped; loadable for CPU evaluation)
<run>/evaluations/%09d/result.json      diagnostic evaluation reports
<run>/evaluations/%09d/<game>.rep       saved native replays (BOPLREP2)
<run>/evaluations/%09d/<game>.json      physics frames for saved games
<run>/evaluations/%09d/<game>.mp4/.jpg  rendered video + poster (optional)
<run>/final-evaluations/%09d/result.json, validation/evaluations/%09d/result.json (optional)
<run>/video-status.json     optional {status, step, rendered}
```

## config.json (required keys the dashboard uses)

`schema:1`, `algorithm:"recurrent PPO"` (the run listing filters on these two),
`steps`, `learner_steps` (target, nullable), `training_maps`, `eval_maps`,
`episode_ticks`, `telemetry_schema:1`, `frozen_opponents:[{name,path,sha256}]`,
`evaluation_baseline` (nullable), `reward` (dict incl. `kill`, `timeout`).

## status.json (required)

`status` (starting|running|complete|stopped|failed), `pid`, `step`,
`learner_steps`, `elapsed`, `matches`, `failure` (nullable),
`target_steps`, `target_learner_steps`. Liveness is checked via `pid`.

## metrics.jsonl — one record per update

Required (all currently written by `train.py`):

- `step`, `learner_steps`, `update`, `elapsed`, `sps`, `learner_fraction`
- `matches`, `map_matches` ({map_id: completed matches})
- `policy_loss`, `value_loss`, `entropy` (joint, nats), `kl`, `clip_fraction`,
  `gradient_norm`, `explained_variance`, `mean_reward`
- `action_rates` ({jump,dash,arrow,platform}: fraction of learner decisions)
- `behavior` = `telemetry.summarize` over learner decisions:
  `{schema:1, decisions, seconds, totals:{dash_activations, bow_activations,
  platform_activations, arrows_fired, arrow_hits, deaths_environment,
  deaths_opponent, deaths_self, deaths_other, reward_outcome, reward_kills,
  reward_deaths, episodes, wins, timeouts, ticks}}`
- `pool_steps`, `opponent_game_decisions`, `script_assignments`
- `cuda_memory_mb`, `rollout_seconds`, `train_seconds`

New — the dashboard renders these once present (per update):

- `entropy_aim` — mean policy entropy of the 256-way aim head, nats
  (max ln 256 = 5.545). Same averaging scope as `entropy` (learner samples
  across all epochs/minibatches of the update, same as the joint number).
- `entropy_buttons` — mean entropy of the 16-way button head, nats
  (max ln 16 = 2.773).
- `entropy_coefficient` — the coefficient actually applied this update
  (already written by the null-hypothesis branch; keep the name).
- `learning_rate` — the optimizer learning rate actually applied this update.
- `advantage_std` — standard deviation of GAE advantages for learner samples
  **before** any normalization, once per update over the whole rollout.
  (Rationale: sparse reward + per-minibatch normalization makes "no signal"
  gradients look full-size; this is the direct detector.)
- `behavior.totals.kills` — kills credited to the learning player by the game
  (opponent deaths the learner caused). Today only `reward_kills` exists and it
  becomes unusable when the kill reward is 0.

## Evaluation result.json (required)

Written by `evaluate.py`; keep this shape: top level `schema:1`, `step`,
`suite`, `frozen_opponents`, `training_maps`, `groups` (per opponent×map:
`opponent, map, games, wins, win_rate, interval` (Wilson), `timeouts`,
`behavior_totals`, `mean_duration`), and `games` — per game:
`opponent, map, seed, side, save, name, win, timeout, kills, deaths, ticks,
behavior, action_rates`, plus for saved games `replay`, `replay_sha256`,
`visual`, `final_players`. The replay browser needs `replay`, `replay_sha256`,
`ticks`, `final_players`, `side`, `opponent`, `map`, `seed`.

The saved-game physics file `<game>.json` keeps
`{schema:1, frames:[{t, players, slimes, entities, actions:[[angle,buttons] x2]}],
tick_rate:60, decision_rate:15}`; `frames[i].actions` (both players' sampled
actions at that decision) powers the human-comparison gauges and heatmaps, so
the native evaluator must keep writing it.

## New: per-decision policy dump for saved evaluation games

For each saved game, write `<game>.mind.json.gz` next to the `.rep`:

```json
{"schema": 1, "step": 123456, "side": 0, "decision_ticks": 4,
 "aim_bins": 256,
 "decisions": [
   {"t": 240, "value": 0.12,
    "buttons": [16 floats, probabilities],
    "aim": [aim_bins floats, probabilities],
    "action": [aim_index, buttons_index]}
 ]}
```

`t` is the game tick the observation was taken at; `value` is V(s) from the
learner's critic; probabilities are post-softmax. `aim_bins` may be 256 or a
uniform downsample to 64. Floats may be rounded to 4 significant digits. The
dashboard shows V(s) and the action distribution under the match video,
synchronized to playback. `research/dump_decisions.py` produces this for
old-format checkpoints by re-simulating saved actions; the native trainer's
evaluator should emit it directly during saved games.

## Checkpoints and the ELO ladder

`research/elo_ladder.py` plays round-robin pools among recent checkpoints plus
fixed anchors (random, scripted opponents, a frozen reference checkpoint). It
needs: the `checkpoints/policy-%09d.pt` naming to survive, and every checkpoint
to be loadable for CPU inference (today `policy.load()`; if the native format
changes, provide an equivalent single-observation CPU step interface and tell
us). Ladder output lives in `<run>/ladder/` and is dashboard-read-only.

## Asks that are trainer configuration (not dashboard code)

- Raise the episode cap so real games can finish: `--episode-ticks` default
  3600 → 7200 (2 min) and the native `max_episode_ticks` to match. Bopl has no
  round timer; 60 s truncates real games. The dashboard now shows truncations
  per episode prominently, so we can revisit the number with data.
- Timeout/truncation semantics must stay visible: keep `behavior.totals.timeouts`
  and `episodes`, and keep `reward.timeout` in `config.reward`.
