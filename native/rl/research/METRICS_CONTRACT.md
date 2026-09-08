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
<run>/checkpoints/policy-%09d.pt                 legacy torch checkpoints, or
<run>/checkpoints/policy-%09d.bin + .bin.json    native checkpoints
<run>/evaluations/%09d/result.json      diagnostic evaluation reports
<run>/evaluations/%09d/<game>.rep       saved native replays (BOPLREP2)
<run>/evaluations/%09d/<game>.json      physics frames for saved games
<run>/evaluations/%09d/<game>.mind.json.gz  per-decision policy dump (see below)
<run>/evaluations/%09d/<game>.mp4/.jpg  rendered video + poster (optional)
<run>/final-evaluations/%09d/result.json, validation/evaluations/%09d/result.json (optional)
<run>/video-status.json     optional {status, step, rendered}
```

Checkpoint loading: only native `.bin` checkpoints are executable, via
`research/native_policy.py load(path) -> policy, meta` with
`policy.initial(batch, 'cpu')` and `policy.step(obs, state) -> logits, values,
state`. Legacy `.pt` files are display-only artifacts (listed, never loaded —
the old torch model classes are deleted and must not be revived). Tools that
discover checkpoints must glob both patterns for listing, `.bin` for play.

## config.json (required keys the dashboard uses)

`schema:1`, `algorithm:"recurrent PPO"` (the run listing filters on these two),
`steps`, `learner_steps` (target, nullable), `training_maps`, `eval_maps`
(legacy) or `evaluation_maps` + `heldout_maps` (native; the dashboard combines
them), `episode_ticks`, `telemetry_schema:1`,
`frozen_opponents:[{name,path,sha256}]`, `evaluation_baseline` (nullable),
`reward` (dict incl. `kill`, `timeout`).

## status.json (required)

`status` (starting|running|complete|stopped|failed), `pid`, `step`,
`learner_steps`, `elapsed`, `matches`, `failure` (nullable or absent),
`target_steps`, `target_learner_steps`. Liveness is checked via `pid`.

## metrics.jsonl — one record per update

Required (all currently written by `train.py`):

- `step`, `learner_steps`, `update`, `elapsed`, `sps`, `learner_fraction`.
  Native semantics (`step_unit: "physical_player_decisions"`): `step` counts
  actual all-player decisions, `learner_steps` counts actually-learned rows,
  `sps` is physical decisions/s, `learner_sps` is reported directly (readers
  prefer it over `sps * learner_fraction`), `learner_fraction` is measured, and
  padded `native_agent_steps` stays a separate field the dashboard ignores.
- `matches`, `map_matches` ({map_id: completed matches})
- `policy_loss`, `value_loss`, `entropy` (joint, nats), `kl`, `clip_fraction`,
  `gradient_norm`, `explained_variance`, `mean_reward`
- `action_rates` ({jump,dash,arrow,platform}: fraction of learner decisions)
- `behavior` — accumulated from learner decision telemetry directly, as
  dedicated cumulative counters that include unfinished episodes. Episode-log
  derived numbers (e.g. Puffer's `static_vec_log`, which drops unfinished
  episodes and clears env logs once any `n>0`) are not acceptable here; only
  `episodes`, `wins` and `timeouts` count completed episodes. Shape matches
  `telemetry.summarize` over learner decisions:
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

Two accepted schemas: `schema:1` (historical runs) and `schema:2` (native
evaluator; adds immutable checkpoint/opponent hashes and per-game explicit
`kills`, `deaths`, `mind`). Readers must support both.

Written today by `evaluate.py`; keep this shape: top level `schema`, `step`,
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
actions) powers the human-comparison gauges and heatmaps, so the native
evaluator must keep writing it. Action timing: legacy files record the action
that led *into* a frame (`frames[0]` has none); native files set top-level
`actions_timing: "observation-before-action"`, meaning `frames[i].actions` is
the action taken *from* frame i and the terminal frame has none. Readers use
the tag to align actions, decisions and mind records. Action encoding: the
aim byte's index 0 means "no joystick direction"; 1..255 map to
`(index-1)*360/255` degrees. The aim-change gauge only compares consecutive
decisions that both aim somewhere.

## New: per-decision policy dump for saved evaluation games

For each saved game, write `<game>.mind.json.gz` next to the `.rep`:

```json
{"schema": 1, "step": 123456, "side": 0, "decision_ticks": 4,
 "aim_bins": 256, "checkpoint_identity": "…",
 "decisions": [
   {"t": 0, "value": 0.12,
    "buttons": [16 floats, probabilities],
    "aim": [aim_bins floats, probabilities],
    "action": [aim_index, buttons_index]}
 ]}
```

`t` is the game tick of the action decision; note the policy's world
observation is delayed 9 ticks (own action-history is immediate), which the
native evaluator labels in metadata — keep that in mind when reading V(s).
`value` is V(s) from the learner's critic; probabilities are post-softmax.
Aim encoding: index 0 is "no joystick direction" (rendered apart, in gray);
indices 1..255 map to `(index-1)*360/255` degrees. `aim_bins` may be 256 or
a uniform downsample to 64. Floats may be rounded to 4 significant digits.
`decisions[i]` aligns with the i-th decision of the game (with
`actions_timing: observation-before-action`, that is `frames[i]`).
Optional per-decision extras the dashboard will render when present:
`action_probability` (probability of the sampled joint action) and
`entropy: {aim, buttons}` (nats). Unknown fields are ignored. Full
distributions stay in this gzipped sidecar rather than embedded in the visual
JSON: 256+16 floats per decision would grow an uncompressed 1800-decision
replay fetch by megabytes, and the replay browser loads visuals far more often
than the mind strip needs distributions. A compact per-frame
`decision: {index, value}` in visual schema 2 is fine as a convenience, but
the sidecar remains the transport of record. The
dashboard shows V(s) and the action distribution under the match video,
synchronized to playback. Implemented natively (native-evaluation rift commit
037a9c8, fixture
`native/artifacts/rl/native-migration-audit/eval-mind-smoke/evaluations/000000000`).
Mind dumps are produced only by the native evaluator; historical games without
one simply don't show the panel (no torch backfill).

## Checkpoints and the ELO ladder

`research/elo_ladder.py` plays round-robin pools among recent checkpoints plus
fixed anchors (random, scripted opponents, a frozen reference checkpoint). Only
native `.bin` checkpoints are playable (via `native_policy.load()`); `.pt`
checkpoints appear in listings but never as ladder participants, so the
"best old checkpoint" anchor is dropped: the old torch GRU (standard recurrent
matrices/biases, Tanh encoder) has no faithful export to the native
MinGRU/bias-free layout. Anchors are random, scripted bots and frozen native
reference checkpoints. Ladder
output lives in `<run>/ladder/ladder.json` (dashboard-read-only):

```json
{"schema": 1, "updated": 0.0, "games": 980, "games_per_pair": 8, "maps": [0],
 "seed_base": 50001, "episode_ticks": 7200, "note": "…",
 "ratings": [{"name": "checkpoint-000000000", "kind": "checkpoint|anchor",
              "step": 0, "sha256": "…", "rating": 0.0,
              "games": 0, "wins": 0, "losses": 0, "draws": 0}],
 "matches": [{"a": "…", "b": "…", "wins_a": 0, "wins_b": 0, "draws": 0}]}
```

Ratings are a Bradley-Terry MLE with timeouts as draws and the random anchor
pinned at 0 Elo.

## Asks that are trainer configuration (not dashboard code)

- Raise the episode cap so real games can finish: `--episode-ticks` default
  3600 → 7200 (2 min) and the native `max_episode_ticks` to match. Bopl has no
  round timer; 60 s truncates real games. The dashboard now shows truncations
  per episode prominently, so we can revisit the number with data.
- Timeout/truncation semantics must stay visible: keep `behavior.totals.timeouts`
  and `episodes`, and keep `reward.timeout` in `config.reward`.
