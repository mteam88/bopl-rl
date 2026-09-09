# Hyperparameter / reward sweeps with PufferLib's Protein

Runs many short `native/rl/puffer/train.py` trials instead of one long run at a
time, searching hyperparameters and reward settings the way PufferLib does.
`train.py` stays the only trainer; the driver launches it as subprocesses and
talks to `pufferlib.sweep.Protein` through its two-call API (`suggest` /
`observe`).

## How Protein works (two paragraphs)

Protein is a cost-aware Bayesian search. Every finished trial contributes not
one number but a downsampled curve of (cost, score) points — cost is learner
decisions (or wall-clock), score is an external yardstick — so the optimizer
learns *score as a function of both hyperparameters and budget*. Two small
Gaussian processes (score and log-cost, on CPU here — `use_gpu = false` — the
training GPU is shared) are fit over the observations; candidate suggestions
are sampled around the current Pareto front of cost vs score, scored by the
GPs, and weighted toward a rotating target cost so ~80% of trials stay cheap
and ~20% expand toward the cost cap (`max_suggestion_cost`). The first trial
always runs the config defaults; the next `num_random_samples` (10) are a
scrambled Sobol sequence, then the GP takes over. Because whole curves are
observed, one lucky seed at a huge budget cannot dominate: the front must be
consistent across costs.

Trials that lag are killed early: Protein fits a robust quantile regression of
score against log cost over everything seen so far (`early_stop_quantile`, 0.3
= the 30th percentile line) and once a running trial is past 20% of its budget
(and the model has ≥30 observations), a score below that line at the matching
cost gets a SIGTERM — `train.py` saves a checkpoint and exits cleanly, and the
partial curve is still observed. `Random` and `ParetoGenetic` are drop-in
alternative methods (same config) for baselines and pure Pareto expansion.

## Layout

```
native/rl/sweep/
  driver.py        the sweep loop (this is what you run)
  sweep_config.py  TOML -> PufferLib sweep_config + train.py flag rendering
  scoring.py       scorers: run directory -> (learner_steps, score) curve
  report.py        results.jsonl, summary.md, launch-best.sh
  configs/         one TOML per sweep
  tests/           unittest suite + fixtures copied from run 007
native/artifacts/rl/sweeps/<name>/
  sweep.json       every suggestion + observation (the resumable state)
  results.jsonl    one line per trial, regenerated after every trial
  summary.md       table + Pareto front + best config
  launch-best.sh   ready-to-run script for the best configuration
  000/ 001/ ...    ordinary train.py run directories (dashboard-compatible)
  000.log ...      trial stdout (train.py JSON progress lines)
```

## Config format

One TOML per sweep, see `configs/smoke.toml` (tiny validation sweep) and
`configs/arrows-and-rewards.toml` (first real sweep, with its budget math).

- `[sweep]` — `method` (Protein | Random | ParetoGenetic), `goal`,
  `metric_distribution` (`percentile` logit-transforms win-rate-like scores),
  `downsample` (points observed per trial curve), `early_stop_quantile`,
  `max_suggestion_cost` (in cost units), `max_runs` (successful trials),
  `use_gpu` (Protein's GP; default false), `cost` (`learner_steps` default,
  or `wall_clock` — learner steps are robust to box load), `seeds`
  (`fixed` | `increment`), `concurrency` (default 1; the box is shared).
- `[trial]` — `memory_max` (systemd MemoryMax per trial), `nice`,
  `max_seconds` (wall cap, driver SIGTERMs past it), `poll_seconds`,
  `min_early_stop_fraction`.
- `[scoring]` — see below.
- `[fixed]` — every fixed `train.py` argument, snake_case
  (`learning_rate = 3e-4` renders as `--learning-rate 0.0003`). `steps` is
  required: it is the default budget and the seed trial's cost.
- `[params.<name>]` — swept parameters as PufferLib spaces: `distribution`
  (`uniform`, `int_uniform`, `uniform_pow2`, `log_normal`, `logit_normal`),
  `min`, `max`, `scale` (`auto` | `time` | number), optional `integer = true`
  (rounded at flag-render time, so Protein observes exactly what it
  suggested) and `default =` (when the default is not in `[fixed]`).
  Names map to `train.py` flags; `head_scale_aim`/`head_scale_buttons`/
  `head_scale_hold` are virtual components composed into
  `--entropy-head-scale`. `steps` is the cost parameter (Protein's
  `train/total_timesteps` analogue); sweep it to give Protein its cost
  dimension. When entropy sweeps below the fixed floor, `--entropy-floor` is
  capped at entropy/2 so trials don't fail validation.

## Launching, resuming, dry runs

```sh
# print the next 3 trial commands without launching or persisting anything
native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
  --config native/rl/sweep/configs/smoke.toml --dry-run 3

# run (survives kill -TERM; re-run the same command to resume)
native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
  --config native/rl/sweep/configs/smoke.toml
```

Each trial runs in its own systemd scope
(`systemd-run --user --scope -p MemoryMax=... --unit bopl-sweep-<sweep>-<id>`)
under `nice`, so a runaway trial cannot starve a real training run. The driver
persists every suggestion and observation in `sweep.json` *before and after*
each trial; killing the driver loses nothing — on restart it re-observes the
whole history into a fresh Protein, adopts any still-running trial (or
finalizes a dead one from its run directory), and continues until `max_runs`
successful trials. A pid lock in `sweep.json` stops a second driver from
racing the same sweep (a dead owner's lock is taken over silently). Failures (no scoreable output) are observed as failures and
don't count toward `max_runs` (attempts are capped at twice `max_runs`).

The driver resolves PufferLib from this checkout's
`native/artifacts/rl/dependencies/PufferLib-*` via `PYTHONPATH` (the venv's
editable install may point at another checkout).

## Scoring

The score is an external yardstick, never the training-time self-play win
rate (self-play sits at ~0.5 by construction). Scorers read the trial's run
directory; the curve is downsampled to `downsample` points along learner
steps (bin means, exact final point) and each point is fed to
`Protein.observe` with the cost at that point — exactly PufferLib's loop.

- `scorer = "anchors"` (default): mean win rate over the run's own periodic
  evaluations (`evaluations/*/result.json`) against fixed opponents —
  `random`, `initial`, plus whatever frozen checkpoints the run's
  `--evaluation-opponents` manifest contributes. The group list is
  data-driven from the results (anything but `selfplay`), so new anchors
  (e.g. scripted bots) join without code changes. Timeouts already count as
  losses. `opponents = [...]` restricts the list; `include_heldout = true`
  adds held-out-map groups.
- `scorer = "behaviour"`: a rate from `metrics.jsonl` per-update behaviour
  totals — `metric = "kills_per_episode" | "arrow_hit_rate" |
  "arrows_per_episode" | "wins_per_episode"`. Note the totals in
  `metrics.jsonl` are per-update windows (not cumulative); curve points sum
  windows.
- `scorer = "match"`: head-to-head of the trial's final checkpoint against a
  fixed `enemy` checkpoint (`games`, `maps`, `workers`), played with the
  native CPU eval library (`research/native_policy.py` + `environment.py`,
  as `elo_ladder.py` does). One observation at final cost; draws/timeouts
  score 0.5. Match results land in `<trial>/match/result.json`.

## Reading results

`summary.md` has the trial table (swept hypers, cost, score, early-stop
reason, Pareto membership), the cost-vs-score Pareto front, and the best
configuration; `launch-best.sh` is a ready-to-run launch script for it
(`STEPS=... OUT=... bash launch-best.sh`, long-run conventions). Raw
per-trial records live in `results.jsonl` and `sweep.json`.

Trials are ordinary run directories, so the dashboard renders them; register
a sweep with the glob option:

```sh
native/artifacts/rl/venv/bin/python native/rl/research/serve.py --port 8876 \
  --run-root-glob 'smoke=native/artifacts/rl/sweeps/smoke/[0-9]*'
```

New trials appear as `smoke-000`, `smoke-001`, ... without restarting the
server (the glob is re-evaluated per request).

## Cautions

- One GPU: keep `concurrency = 1` while anything else trains on the box, and
  never launch a sweep alongside a run you can't afford to slow down.
- Protein's GP stays on CPU by default (`use_gpu = false`); it is tiny.
- gamma has no `train.py` flag (native default); add a flag before sweeping it.
- Validation-scale trials (a few million steps) are noisy; treat smoke sweeps
  as plumbing checks, not science.
