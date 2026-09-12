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

Protein can also early-stop lagging trials against a quantile regression of
score vs log cost (`early_stop_quantile`); the driver does not use it, because
the reference scorer plays its games after a trial ends, so only the wall cap
(`[trial].max_seconds`) SIGTERMs a trial — `train.py` saves a checkpoint and
exits cleanly. `Random` and `ParetoGenetic` are drop-in alternative methods
(same config) for baselines and pure Pareto expansion.

## Layout

```
native/rl/sweep/
  driver.py        the sweep loop (this is what you run)
  sweep_config.py  TOML -> PufferLib sweep_config + train.py flag rendering
  scoring.py       run directory -> (learner_steps, score) curve via the reference pool
  reference_pool.py  frozen pool: build / verify / rate / rate-runs / human
  flags.py         behaviour flags from evaluation behaviour totals (labels only)
  report.py        results.jsonl, summary.md, launch-best.sh
  configs/         one TOML per sweep (smoke.toml = the pipeline at a tiny budget;
                   overnight-3.toml is a historical record and no longer loads)
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

One TOML per sweep, see `configs/smoke.toml` (the live pipeline at a tiny
budget) and `configs/overnight-4.toml` (the current sweep, with its budget
math in the header).

- `[sweep]` — `method` (Protein | Random | ParetoGenetic), `goal`,
  `metric_distribution` (`percentile` logit-transforms win-rate-like scores),
  `downsample` (points observed per trial curve), `early_stop_quantile`
  (passed to Protein, unused by the driver), `max_suggestion_cost` (in cost units), `max_runs` (successful trials),
  `num_random_samples` (Protein: Sobol trials after the defaults trial before
  the GP suggests; PufferLib's default 10 is half of a 16-trial overnight),
  `use_gpu` (Protein's GP; default false), `cost` (`learner_steps` default,
  or `wall_clock` — learner steps are robust to box load), `seeds`
  (`fixed` | `increment`), `concurrency` (default 1; the box is shared).
- `[trial]` — `memory_max` (systemd MemoryMax per trial), `nice`,
  `max_seconds` (wall cap, driver SIGTERMs past it), `poll_seconds`.
- `[scoring]` — `scorer = "reference"` and its settings, see below.
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

# plumbing check (one ~3 min trial, reference scorer, flags), outside the artifacts tree
native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
  --config native/rl/sweep/configs/smoke.toml --root /tmp/sweeps

# the real thing, as a user unit (survives kill -TERM; re-run to resume)
systemd-run --user --unit bopl-sweep-overnight-4 -p MemoryMax=4G -p CPUQuota=900% --nice=10 \
  --working-directory="$PWD" bash -c "native/artifacts/rl/venv/bin/python \
  native/rl/sweep/driver.py --config native/rl/sweep/configs/overnight-4.toml"
```

The driver runs on the Rust sim (`BOPL_RL_BACKEND=rust` is its default): the
trials through `train.py`, the reference scorer's games through
`reference_pool.py`.

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
rate (self-play sits at ~0.5 by construction). The scorer reads the trial's
run directory once the trial has finished; the curve is downsampled to
`downsample` points along learner steps (bin means, exact final point) and
each point is fed to `Protein.observe` with the cost at that point — exactly
PufferLib's loop.

`scorer = "reference"`: Bradley–Terry Elo against a frozen reference pool
(next section). `pool = "<path>/pool.json"` (relative to the checkout),
`last_k = 3` evaluations aggregated by IQM, `games = 240` per checkpoint,
`first_pass_games = 6` per member, `band = [0.15, 0.85]`, `round_games = 48`,
`workers = 4`, `seed_base = 70001`. Elo is unbounded, so
`metric_distribution = "linear"` is required (`percentile` logit-transforms a
0..1 rate). Ratings are cached in `<trial>/reference/<checkpoint>.json`
(rating, 95% CI, per-member records, allocation rounds).

### Why a reference pool, and why it is frozen

overnight-1..3 were scored by an `anchors` scorer (removed 2026-09-11): the
mean win rate of each periodic evaluation over eight hand-picked opponents.
It measured ±3–4 points of standard error per evaluation on overnight-3 while
trial scores wandered 0.27–0.51 within one run; `random`/`initial` groups were
100% wins for every trial and only diluted the mean; the run007 checkpoints
were trained on an older sim with an exploitable platform bug and inflated
new-sim agents; the top-scoring trial (001) fired zero arrows; and re-scoring
25 finals against pool v3 reordered 24 of 25 ranks
(`sweeps/overnight-3-rescore-v5/summary.md`). Those sweeps' `results.jsonl`
keep their anchors scores. The reference scorer follows the literature
instead:

- OpenAI Five, appendix J: rate agents by TrueSkill/Elo against a *frozen*
  pool of reference agents whose ratings are held constant, play mostly the
  references inside the 15–85% expected-win band (lopsided games carry no
  information), add references as agents outgrow the pool, never mutate one.
- Balduzzi et al. 2018, *Re-evaluating evaluation*: a mean over a
  hand-picked pool is biased by redundant and weak entries; a rating over the
  whole tournament graph is not.
- *Elo Uncovered*: sequential Elo depends on game order and K; fit batch
  Bradley–Terry MLE instead (`elo_ladder.bradley_terry`, draws = half wins,
  one virtual draw per pair, random = 0 Elo).
- Agarwal et al. 2021, *rliable*: report the interquartile mean with
  bootstrap CIs rather than a single point (here IQM over the last k
  evaluations; with k < 4 the trim is empty and IQM equals the mean).

`native/rl/sweep/reference_pool.py` implements it:

```sh
# build: full round robin (both sides, evaluation maps, fixed seeds,
# 48 games per pair), BT-MLE fit, pool.json with every checkpoint hash.
# Games involving rollout-search bots (bopl_rl_fork) run in their own, wider
# thread pool and are timed separately (pool.json `timing`). Pool v5 (the
# current one; v1-v4 are deleted):
BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/sweep/reference_pool.py build \
  --out native/artifacts/rl/sweeps/anchors/pool-v5/pool.json \
  --scripted archer-hard,pursuer-hard,evasive-hard,mixed-hard,mixed-medium,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard,mixed-v2-hard,search-hard,search-medium \
  --checkpoint o3-019-final=native/artifacts/rl/sweeps/anchors/overnight-3-v3/o3-019-final.bin ... \
  --games-per-pair 48 --workers 10 --slow-workers 12
# verify: re-hash every checkpoint and the pool itself
native/artifacts/rl/venv/bin/python native/rl/sweep/reference_pool.py verify .../pool.json
# rate one checkpoint (writes <run>/reference/<pool_sha8>/<stem>.json when the
# file lives under checkpoints/; otherwise pass --out)
BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/sweep/reference_pool.py rate \
  --pool .../pool.json --checkpoint <run>/checkpoints/policy-000266698752.bin
```

`pool.json` records each member's kind (`random`, `scripted`, `checkpoint`),
rating, W-L-D, the full match matrix, the sim it was played on (path and
SHA-256), and for checkpoints the weights and sidecar hashes; `pool_sha256`
hashes the sorted (name, kind, sha256, rating) rows. Every load re-hashes the
checkpoints and recomputes `pool_sha256`, so an edited rating or a swapped
weights file fails loudly.

**Adding a reference**: copy the checkpoint (and its `.bin.json` sidecar)
next to the pool, e.g. `native/artifacts/rl/sweeps/anchors/<pool>/`, and
build a *new* pool file (`--out .../pool-2.json`) that lists the old members
plus the new one; point the next sweep's `[scoring].pool` at it. The pool
ratings are re-fit from a fresh round robin, so all members stay on one
consistent scale; older sweeps keep their own pool file and their ratings
remain interpretable against it. `build` refuses to overwrite an existing
pool.

**Never remove or mutate a member**: a candidate's rating is the MLE of its
games against members with *fixed* ratings. Change a member's weights or
rating and every candidate rated before and after sits on different scales
without any file saying so — that is exactly what the pool hash guards
against, and why `verify` runs on every load. If a member turns out to be a
bad reference (exploitable, redundant), leave it in and let the band
allocation starve it of games: a member far outside 15–85% only ever gets the
first-pass games.

**Sim identity**: `pool.json` stores the sim library's SHA-256 and every
rating file records the sim it ran on plus `game_mismatch`. Ratings played on
a different sim build than the pool are flagged but not refused (a rebuilt
`libbopl_sim.so` would otherwise fail every trial of a running sweep); after
a sim change with gameplay impact, build a new pool.

### Behaviour flags

Every finished trial's evaluations are labelled from their behaviour totals
(`evaluate.py` `behavior_totals`, pooled over the non-selfplay groups on the
evaluation maps) with thresholds from the config:

```toml
[scoring.flags]
no-offense = "arrows_per_game < 0.05"
platform-spam = "platforms_per_game > 8"
stalemate = "timeout_rate > 0.5"
```

Metrics: `arrows_per_game`, `arrow_hits_per_game`, `arrow_hit_rate`,
`platforms_per_game`, `dashes_per_game`, `bows_per_game`, `kills_per_game`,
`deaths_per_game`, `timeout_rate`, `win_rate`, `mean_duration_seconds`,
`games`; operators `< <= > >=`. Flags are written into each evaluation
`result.json` (`flags`, `flag_metrics`, `flag_rules`), the trial
`result.json` (`flags_by_evaluation` too), the trial record in `sweep.json`,
the `flags` column of `summary.md`, and appear as a badge on the dashboard
run page. They never change the score: a camper keeps its rating and its
`no-offense` label side by side.

### Rating files

Rating files are derived data on the run, not a second source of truth:

```
<run>/reference/<pool_sha8>/<checkpoint>.json
```

`pool_sha8` is the first eight hex of `pool_sha256` (v5 is `ec5bd5ae`). A rating
on another pool lives in another directory; the dashboard shows the current
pool's scale and lists runs lacking it as unrated. Cache key is (pool sha,
checkpoint sha, games). A fuller band rating is reused for a cheaper request;
an `evaluate.py` fit (`source: evaluation`) is not.

`evaluate.py` writes the cheap fit from its games against pool members on
pool maps (no extra play). The sweep driver and `rate-runs` write the full
band-allocated rater of the last k evaluation checkpoints. The driver's
score is the IQM of those k ratings.

The overnight-4 driver from `1a4cd3a` still writes
`<run>/reference/<checkpoint>.json` (flat). Both locations are read; new
files only go in the sha8 directory. Optional migration:

```sh
mkdir -p "$RUN/reference/ec5bd5ae"
mv "$RUN"/reference/policy-*.json "$RUN/reference/ec5bd5ae/"
```

### Human Elo

Same fixed-member BT, from `native/artifacts/rl/human/mac/games.jsonl` minus
`excluded.json`. `agent_won` means the bot won. Maps rotate 0–123 while the
pool plays 0–2; n is small. Shown as the human row on the dashboard.

```sh
native/artifacts/rl/venv/bin/python native/rl/sweep/reference_pool.py human \
  --pool native/artifacts/rl/sweeps/anchors/pool-v5/pool.json
```

### Rating old runs

`rate-runs` fills the same `<run>/reference/<pool_sha8>/` directory for any
run or sweep (overnight-1/2, long runs, overnight-3). Older sims set
`game_mismatch`. There is no separate rescore tree.

```sh
BOPL_RL_BACKEND=rust systemd-run --user --scope -p CPUQuota=300% -p MemoryMax=6G --nice=15 -- \
  native/artifacts/rl/venv/bin/python native/rl/sweep/reference_pool.py rate-runs \
  --pool native/artifacts/rl/sweeps/anchors/pool-v5/pool.json \
  --sweep native/artifacts/rl/sweeps/overnight-3 \
  --last-k 3 --games 240 --workers 3
```

## Reading results

`summary.md` has the trial table (swept hypers, cost, score, early-stop
reason, Pareto membership), the cost-vs-score Pareto front, and the best
configuration; `launch-best.sh` is a ready-to-run launch script for it
(`STEPS=... OUT=... bash launch-best.sh`, long-run conventions). Raw
per-trial records live in `results.jsonl` and `sweep.json`.

Trials are ordinary run directories, so the dashboard renders them. Register
roots with `--root`, isolated live runs with `--run-root NAME=PATH`, and a
sweep's trials with `--run-root-glob`. `--pool` is the scale (default
pool-v5). `--ladder` and `--study-root` are gone.

```sh
native/artifacts/rl/venv/bin/python native/rl/research/serve.py \
  --host 127.0.0.1 --port 8876 \
  --root native/artifacts/rl \
  --pool native/artifacts/rl/sweeps/anchors/pool-v5/pool.json \
  --run-root-glob 'overnight-3=native/artifacts/rl/sweeps/overnight-3/[0-9]*' \
  --run-root-glob 'overnight-4=native/artifacts/rl/sweeps/overnight-4/[0-9]*'
```

`/` is the ladder (IQM3 ± CI on pool-v5, pool members frozen, human row).
`/run/<name>` is Elo over steps plus win rates, behaviour, checkpoints, replay.
`/sweep/<name>` is the trial table and Pareto front. New glob matches appear
without restarting.

## Cautions

- One GPU: keep `concurrency = 1` while anything else trains on the box, and
  never launch a sweep alongside a run you can't afford to slow down.
- Protein's GP stays on CPU by default (`use_gpu = false`); it is tiny.
- gamma has no `train.py` flag (native default); add a flag before sweeping it.
- Validation-scale trials (a few million steps) are noisy; treat smoke sweeps
  as plumbing checks, not science.
