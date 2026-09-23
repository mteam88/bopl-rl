# Dashboard metrics contract

What the research dashboard (`serve.py`, `web/`) reads from a
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
<run>/checkpoints/policy-%09d.bin + .bin.json    native checkpoints
<run>/checkpoints/policy-%09d.pt                 historical runs only (read-only)
<run>/evaluations/%09d/result.json      diagnostic evaluation reports
<run>/evaluations/%09d/<game>.rep       saved native replays (BOPLREP2)
<run>/evaluations/%09d/<game>.json      physics frames for saved games
<run>/evaluations/%09d/<game>.mind.json.gz  per-decision policy dump (see below)
<run>/evaluations/%09d/<game>.mp4/.jpg  rendered video + poster (optional)
<run>/final-evaluations/%09d/result.json, validation/evaluations/%09d/result.json (optional)
<run>/reference/<pool_sha8>/<checkpoint>.json   derived Elo vs the current pool (evaluate.py + rate-runs)
<run>/reference/<checkpoint>.json               legacy overnight-4 location, still read
<run>/video-status.json     optional {status, step, rendered}
```

Checkpoint loading: native `.bin` is the only checkpoint format, loaded via
`research/native_policy.py load(path) -> policy, meta` with
`policy.initial(batch, 'cpu')` and `policy.step(obs, state) -> logits, values,
state`. New runs never write `.pt`; the files exist only in prior experiments,
where the dashboard lists and serves them as read-only evidence (the old torch
model classes are deleted and must not be revived to load them).

## config.json (required keys the dashboard uses)

`schema:1`, `algorithm:"recurrent PPO"` (the run listing filters on these two),
`steps`, `learner_steps` (target, nullable), `training_maps`,
`evaluation_maps` + `heldout_maps` (the dashboard combines them; historical
runs wrote a combined `eval_maps` instead, still read), `episode_ticks`,
`telemetry_schema:1`,
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
- `matches`, `map_matches` ({map_id: completed matches}), cumulative since run start.
- `policy_loss`, `value_loss`, `entropy` (joint, nats), `kl`, `clip_fraction`,
  `gradient_norm`, `explained_variance`, `mean_reward`
- `action_rates` ({jump,dash,arrow,platform}: fraction of learner decisions)
- `behavior` — accumulated from learner decision telemetry directly, as
  dedicated counters that include unfinished episodes. Each metrics row reports
  the delta for that update, not the lifetime total. Episode-log
  derived numbers (e.g. Puffer's `static_vec_log`, which drops unfinished
  episodes and clears env logs once any `n>0`) are not acceptable here; only
  `episodes`, `wins` and `timeouts` count completed episodes. Shape matches
  `telemetry.summarize` over learner decisions:
  `{schema:1, decisions, seconds, totals:{dash_activations, bow_activations,
  platform_activations, arrows_fired, arrow_hits, deaths_environment,
  deaths_opponent, deaths_self, deaths_other, reward_outcome, reward_kills,
  reward_deaths, episodes, wins, timeouts, ticks}}`
- `pool_steps` (physical decisions of the snapshot each frozen bank currently
  serves, sorted), `opponent_game_decisions`, `script_assignments`
- `cuda_memory_mb`, `rollout_seconds`, `train_seconds`
- `pool` — PufferLib self-play pool state after this update:
  `{schema:1, elo (learner), size, winrate (learner vs all banks since their
  last swaps, nullable), entries:[{step, learner_steps, elo, games, winrate}],
  banks:[{opponent_step, opponent_elo, games, winrate, pending_step,
  winrate_at_swap, epochs_to_align}]}`. `games`/`winrate` on entries are
  lifetime totals for that snapshot as an opponent; on banks they cover the
  current opponent only. `<run>/pool.jsonl` holds the corresponding events
  (`snapshot`, `evicted` with lifetime stats, `armed`, `loaded` with the
  retired opponent's stats), each with `learner_steps` and the learner `elo`.
  With `--scripted-bots` / `--scripted-roster`, `pool.banks` covers only the
  policy banks (or all banks when leftover frozen matches are possible);
  the scripted banks report per update under `scripted` and the sibling
  fields `scripted_share`, `scripted_learner_elo`:
  `scripted` is `{"<kind>-<level>": {bank, elo, games, winrate, weight,
  window_games, window_wins}}` — `games`/`winrate`/`elo` are lifetime
  decided learner-vs-entry totals (timeouts excluded); `window_*` and
  `weight` are this update's draw. `scripted_share` is the bot-match
  fraction pushed to C (`clamp(share_min, share_max * I)` with
  `I = max 4p(1-p)`). `scripted_learner_elo` is the pool's learner Elo
  used for matchmaking. (`native/rl/SCRIPTED_BOTS.md`, `--scripted-roster`).

New — the dashboard renders these once present (per update):

- `entropy_aim` — mean policy entropy of the 256-way aim head, nats
  (max ln 256 = 5.545). Same averaging scope as `entropy` (learner samples
  across all epochs/minibatches of the update, same as the joint number).
- `entropy_buttons` — mean entropy of the 16-way button head, nats
  (max ln 16 = 2.773).
- `entropy_coefficient` — the coefficient actually applied this update
  (already written by the null-hypothesis branch; keep the name).
- `learning_rate` — the optimizer learning rate actually applied this update.
- `advantage_std` — standard deviation of raw GAE advantages for learner
  rollout samples, before normalization. Native `bopl_adv_stats` captures
  population variances once per update before the first minibatch optimizer
  step; frozen rows are excluded.
- `explained_variance` — native `1 - var(A)/var(A+V)` over the same learner
  samples, with raw GAE advantages `A` and collected rollout values `V`.
  Emitted when both variances are finite and `var(A+V) > 0`. This is measured
  before training on the rollout, whereas `value_loss` averages training
  minibatches. For each valid record with ev < 1, the target variance can be
  recovered as `advantage_std**2 / (1-explained_variance)`; aggregate after
  this calculation. Neither metric alone establishes stronger play.
- `behavior.totals.arrow_hits` — arrow contacts that killed another player,
  credited to learner players. `PlayerCollision` calls `RlTelemetry.ArrowHit`
  only for `killed && arrowForTelemetry`; owner hits are excluded. In the
  two-player, stocks-0 Dash/Bow/Platform setup, an arrow can be counted at
  most once for killing its sole opponent. The counter does not track
  individual projectile identities or nonlethal contacts.
- `behavior.totals.wins` — decided wins only. The game awards `WonThisRound`
  to every side eliminated on the final tick, so a round it left undecided
  carries the win flag for all of them; those episodes count in `episodes`,
  never in `wins`, and pay the loss outcome to everyone in them. `wins` and
  `timeouts` therefore do not partition `episodes`.
- `behavior.totals.kills` — kills of somebody else, for learner players.
  `Player.Kill` increments the credited player's `Kills` even when killer
  equals victim, so the game's counter includes self-kills; training subtracts
  `deaths_self` back out, which leaves exactly the kills earned against another
  player. Suicides remain visible in `deaths_self`. `reward_kills` is this same
  quantity scaled by the kill coefficient. Learner-only totals need not equal
  `deaths_self + deaths_opponent`, since kills and deaths concern different
  players and frozen opponents are excluded from learner totals. Per-game
  evaluation `kills` and group `mean_kills` make the same subtraction.

## Evaluation result.json (required)

Two accepted schemas: `schema:1` (historical runs) and `schema:2` (native
evaluator; adds immutable checkpoint/opponent hashes and per-game explicit
`kills`, `deaths`, `mind`). Readers must support both.

Written today by `evaluate.py`; keep this shape: top level `schema`, `step`,
`suite`, `frozen_opponents`, `scripted_opponents` (list of scripted bot names
like `mixed-hard`, scheduled on evaluation and held-out maps alongside
`random`; see `native/rl/SCRIPTED_BOTS.md`), `training_maps`, `groups` (per opponent×map:
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
evaluator must keep writing it. Action timing: historical files record the
action that led *into* a frame (`frames[0]` has none); native files set top-level
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

## Checkpoints and the quality numbers

There are two families of numbers. They never share an axis.

**Frozen-pool IQM** is still the sweep objective: Bradley-Terry Elo vs one
frozen `pool.json` (members held fixed, random = 0), summarised as the IQM of
the last three rated checkpoints with a 95% interval. `evaluate.py` and
`rate-runs` write `<run>/reference/<pool_sha8>/<stem>.json`. The dashboard
labels each such curve by the pool directory name (`pool v7 IQM3`, `pool v6 IQM3`,
…) and will not mix pools. The primary label (`--pool`, default pool v7 =
`sweeps/anchors/pool-v7`, the sweep objective) only picks which curve is the
run page's headline `iqm`; the index shows one column per pool that rated a run.

**Joint Elo** is one Bradley-Terry over every pool round-robin `matches[]`,
every on-disk band-rated candidate `records` dict, and human games (excluded
rounds dropped). Identity is checkpoint sha256 / bot name / `random` / `human`.
Regularizer is one virtual draw per played pair so 480-game matches keep
their weight. In-run evaluation fits (`source: evaluation`, ~144 games, no
band matching, mostly against the scripted training roster) are excluded:
in a joint fit they rated mid-run checkpoints 900+ through exploited bots
alone. They still draw the run page's dashed in-run curve. `GET /api/ladder` is this fit; `GET /api/index` lists every
discovered run. A run payload (`GET /api/run/<name>`) carries both `joint`
and labelled `pools[]`. There is no `<run>/ladder/ladder.json`.

## Asks that are trainer configuration (not dashboard code)

- Raise the episode cap so real games can finish: `--episode-ticks` default
  3600 → 7200 (2 min) and the native `max_episode_ticks` to match. Bopl has no
  round timer; 60 s truncates real games. The dashboard now shows truncations
  per episode prominently, so we can revisit the number with data.
- Timeout/truncation semantics must stay visible: keep `behavior.totals.timeouts`
  and `episodes`, and keep `reward.timeout` in `config.reward`.

## Dashboard reader limits

The index, joint ladder, and sweep payloads are assembled from rating files
(`reference/<pool_sha8>/*.json` and legacy `reference/*.json`), each pool's
`matches[]`, `status.json`, `result.json` flags, `sweep.json`, and
`human/mac/games.jsonl` (minus `excluded.json`). They do not parse
`metrics.jsonl` or evaluation `result.json` bodies to fit Elo. Frozen-pool
bootstrap intervals already stored in a rating file are reused; they are not
refit. The joint fit is a new MLE (sparse Newton, diagonal Wald CI) cached by
the mtimes of those inputs. A background thread keeps the last good index,
ladder, and sweep JSON warm; responses are ETag'd.

`GET /api/run/<name>` is the first-paint run payload: status, config, labelled
frozen-pool curves (`pools[]`, `iqm` for the primary pool), the joint curve,
human records against this run's checkpoints when he played them, flags, an
evaluation directory list, and the **latest**
evaluation report (groups + replay metadata). Older evaluation reports load
from `GET /api/run/<name>/evaluation/<dir>`. Training charts load from
`GET /api/run/<name>/metrics` (not the first paint): at most 360 points with
the scalar keys the charts draw. The source JSONL remains complete.

The metrics reader (`metrics_store.py`) still retains at most 2,400 samples
per run (eight recently viewed runs), keeping the latest 120 updates unchanged
and doubling the sampling stride for older history. `metrics_total` counts all
source records; `map_matches` is the latest cumulative counters. A partial
final line waits for the next append. A malformed complete line fails the
metrics request; file replacement or truncation resets the reader.

Large JSON payloads gzip over the wire. The browser keeps a stale-while-revalidate
cache of view JSON and sends `If-None-Match`.

Reader regression check: `python3 native/rl/research/check_metrics_store.py`.
HTTP cache check: `python3 native/rl/research/check_dashboard_api.py`.
