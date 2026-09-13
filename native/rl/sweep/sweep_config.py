"""Sweep configuration: TOML file -> PufferLib sweep_config dict + train.py flags.

One TOML file declares one sweep (see configs/). Sections:

  [sweep]    method/goal/downsample/early_stop_quantile/max_suggestion_cost/
             max_runs/num_random_samples/use_gpu/cost/seeds/concurrency
  [trial]    per-trial resource limits (MemoryMax, wall cap, poll cadence)
  [scoring]  the reference-pool scorer's settings (see scoring.py);
             [scoring.flags] behaviour flag rules (flags.py), labels only
  [fixed]    every fixed train.py argument, snake_case (steps is the budget)
  [params.<name>]  swept parameters as PufferLib spaces:
             distribution = uniform | int_uniform | uniform_pow2 | log_normal |
             logit_normal, with min/max/scale (and optional integer = true,
             default = <value> when the default is not in [fixed])

The cost parameter is `steps` (learner decisions), PufferLib's
`train/total_timesteps` analogue. Declare [params.steps] to let the optimizer
trade cost against score (Protein's intended regime); without it every trial
runs the fixed budget.

This module is pure stdlib so unit tests import it without torch.
"""
import math
import re
import tomllib
from pathlib import Path

import flags

# Reserved keys PufferLib's _params_from_puffer_sweep skips; parameters may not
# use these names (they live in their own [params.*] tables, but the flattened
# sweep_config dict would collide).
RESERVED = ('method', 'metric', 'metric_distribution', 'goal', 'downsample',
            'use_gpu', 'prune_pareto', 'sweep_only', 'max_suggestion_cost',
            'early_stop_quantile', 'gpus', 'max_runs')

DISTRIBUTIONS = ('uniform', 'int_uniform', 'uniform_pow2', 'log_normal', 'logit_normal')

# train.py argparse destinations (native/rl/puffer/train.py arguments()).
# output/resume are driver-owned; seed is driver-owned when [sweep].seeds != fixed.
TRAIN_FLAGS = ('game', 'steps', 'workers', 'buffers', 'banks', 'seed', 'encoder',
               'hidden', 'layers', 'horizon', 'learner_passes', 'learning_rate',
               'anneal_lr', 'min_lr_ratio', 'entropy', 'entropy_floor',
               'entropy_half_life', 'entropy_head_scale', 'entropy_free_rows',
               'episode_ticks', 'reward_kill', 'reward_timeout', 'stocks',
               'training_maps', 'checkpoint_every', 'pool_size', 'swap_winrate',
               'min_games', 'snapshot_interval', 'opponent_timeout', 'elo_init',
               'elo_k', 'pool_seed', 'cudagraphs', 'no_eval', 'eval_every',
               'evaluation_maps', 'heldout_maps', 'evaluation_baseline',
               'evaluation_opponents', 'evaluation_games', 'heldout_games',
               'frozen_games', 'evaluation_workers', 'scripted_bots', 'scripted_level',
               'scripted_roster', 'scripted_share_min', 'scripted_share_max',
               'scripted_opponents', 'aim_fourier', 'aim_ar')
STORE_TRUE = ('no_eval',)
# Virtual swept parameters for the scripted-bot curriculum (SCRIPTED_BOTS.md):
# scripted_banks = how many of [fixed].scripted_bots (an ordered candidate
# list) take over banks, scripted_level_index = easy/medium/hard.
SCRIPTED_LEVELS = ('easy', 'medium', 'hard')
SCRIPTED_VIRTUAL = ('scripted_banks', 'scripted_level_index')
# Virtual swept parameters composed into --entropy-head-scale "aim,buttons,hold".
HEAD_SCALE = {'head_scale_aim': 0, 'head_scale_buttons': 1, 'head_scale_hold': 2}
# Parameters that must be integers on the command line even when the space
# yields floats (log_normal steps, for example).
ALWAYS_INTEGER = ('steps', 'workers', 'buffers', 'banks', 'hidden', 'layers', 'aim_fourier', 'aim_ar',
                  'scripted_banks', 'scripted_level_index',
                  'horizon', 'learner_passes', 'entropy_half_life', 'episode_ticks',
                  'pool_size', 'min_games', 'snapshot_interval', 'opponent_timeout',
                  'checkpoint_every', 'eval_every', 'evaluation_games',
                  'heldout_games', 'frozen_games', 'evaluation_workers', 'seed')


class ConfigError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


class SweepSpec:
    """Parsed sweep configuration."""

    def __init__(self, raw, name):
        self.raw = raw
        self.name = name
        sweep = dict(raw.get('sweep', {}))
        self.method = sweep.get('method', 'Protein')
        _require(self.method in ('Protein', 'Random', 'ParetoGenetic'),
                 f'Unknown sweep method {self.method!r}')
        self.goal = sweep.get('goal', 'maximize')
        self.metric = sweep.get('metric', 'score')
        self.metric_distribution = sweep.get('metric_distribution', 'linear')
        _require(self.metric_distribution in ('linear', 'percentile'),
                 'metric_distribution must be linear or percentile')
        self.downsample = int(sweep.get('downsample', 5))
        self.early_stop_quantile = float(sweep.get('early_stop_quantile', 0.3))
        self.max_suggestion_cost = float(sweep.get('max_suggestion_cost', 3600))
        self.max_runs = int(sweep.get('max_runs', 20))
        # Protein: Sobol trials (after the defaults trial) before the GP suggests.
        self.num_random_samples = int(sweep.get('num_random_samples', 10))
        _require(self.num_random_samples >= 1, 'num_random_samples must be positive')
        # fixed_cost: every trial runs exactly [fixed].steps; Protein still gets
        # a cost axis (the partial curve points) but suggests at the fixed budget.
        self.fixed_cost = bool(sweep.get('fixed_cost', False))
        # The training GPU is shared with real runs; the GP is tiny, keep it on CPU.
        self.use_gpu = bool(sweep.get('use_gpu', False))
        self.cost = sweep.get('cost', 'learner_steps')
        _require(self.cost in ('learner_steps', 'wall_clock'),
                 'cost must be learner_steps or wall_clock')
        self.seeds = sweep.get('seeds', 'fixed')
        _require(self.seeds in ('fixed', 'increment'), 'seeds must be fixed or increment')
        self.concurrency = int(sweep.get('concurrency', 1))
        _require(self.concurrency >= 1, 'concurrency must be positive')

        trial = dict(raw.get('trial', {}))
        self.memory_max = str(trial.get('memory_max', '6G'))
        self.nice = int(trial.get('nice', 10))
        self.max_seconds = float(trial.get('max_seconds', 3600))
        self.poll_seconds = float(trial.get('poll_seconds', 30))
        self.unit_prefix = str(trial.get('unit_prefix', 'bopl-sweep'))

        self.scoring = dict(raw.get('scoring', {}))
        _require(self.scoring.get('scorer') == 'reference',
                 f"[scoring].scorer must be \"reference\", not {self.scoring.get('scorer')!r}")
        _require(isinstance(self.scoring.get('pool'), str), '[scoring].pool (a pool.json path) is required')
        _require(int(self.scoring.get('last_k', 3)) >= 1, '[scoring].last_k must be positive')
        band = self.scoring.get('band', (.15, .85))
        _require(len(band) == 2 and 0 < band[0] < band[1] < 1, '[scoring].band must be [low, high] in (0, 1)')
        _require(self.metric_distribution == 'linear',
                 'scorer = "reference" yields Elo, not a rate: set metric_distribution = "linear"')
        # Behaviour flags (flags.py): labels only, never part of the score.
        self.flag_rules = dict(self.scoring.get('flags', flags.DEFAULT_RULES))
        for name, rule in self.flag_rules.items():
            _require(bool(re.fullmatch(r'[a-z][a-z0-9-]*', name)), f'Flag name {name!r} must be [a-z][a-z0-9-]*')
            try:
                flags.parse_rule(rule)
            except ValueError as error:
                raise ConfigError(str(error)) from None

        self.fixed = dict(raw.get('fixed', {}))
        _require('output' not in self.fixed and 'resume' not in self.fixed,
                 'output/resume are driver-owned; remove them from [fixed]')
        _require('steps' in self.fixed, '[fixed].steps (the default trial budget) is required')
        for key in self.fixed:
            _require(key in TRAIN_FLAGS, f'[fixed].{key} is not a train.py flag')

        self.params = {}
        for pname, spec in dict(raw.get('params', {})).items():
            _require(pname not in RESERVED, f'Parameter name {pname!r} is reserved by PufferLib')
            _require(pname in TRAIN_FLAGS or pname in HEAD_SCALE or pname in SCRIPTED_VIRTUAL,
                     f'[params.{pname}] does not map to a train.py flag')
            _require(isinstance(spec, dict) and 'distribution' in spec,
                     f'[params.{pname}] needs a distribution')
            _require(spec['distribution'] in DISTRIBUTIONS,
                     f'[params.{pname}] distribution must be one of {DISTRIBUTIONS}')
            for key in ('min', 'max'):
                _require(key in spec, f'[params.{pname}] needs {key}')
            spec = dict(spec)
            spec.setdefault('scale', 'auto')
            self.params[pname] = spec
        _require(self.params, 'At least one [params.*] entry is required')
        _require(not (self.fixed_cost and 'steps' in self.params),
                 'fixed_cost sweeps must not sweep steps')
        if any(name in self.params for name in SCRIPTED_VIRTUAL):
            _require(self.scripted_pool(), '[fixed].scripted_bots (ordered candidate kinds) is required '
                     'to sweep scripted_banks/scripted_level_index')

    def scripted_pool(self):
        """Ordered bot kinds from [fixed].scripted_bots (list or comma string)."""
        value = self.fixed.get('scripted_bots', [])
        return [v for v in (value if isinstance(value, list) else str(value).split(',')) if v]

    # ---- Protein / Random / ParetoGenetic construction -----------------------

    def sweep_config(self):
        """The flat dict pufferlib.sweep.Hyperparameters expects."""
        config = dict(method=self.method, metric=self.metric, goal=self.goal,
                      metric_distribution=self.metric_distribution,
                      downsample=self.downsample,
                      early_stop_quantile=self.early_stop_quantile,
                      max_suggestion_cost=self.max_suggestion_cost,
                      use_gpu=self.use_gpu, max_runs=self.max_runs)
        for name, spec in self.params.items():
            config[name] = dict(distribution=spec['distribution'],
                                min=spec['min'], max=spec['max'], scale=spec['scale'])
        if self.fixed_cost:
            steps = int(self.fixed['steps'])
            config['steps'] = dict(distribution='log_normal', min=steps / 16, max=steps, scale='time')
        return config

    def defaults(self):
        """Hyperparameter dict of the first (seed) trial: the config defaults."""
        values = {}
        for name, spec in self.params.items():
            if 'default' in spec:
                value = spec['default']
            elif name in self.fixed:
                value = self.fixed[name]
            elif name in HEAD_SCALE:
                triple = _head_scale_triple(self.fixed.get('entropy_head_scale', '1,1,1'))
                value = triple[HEAD_SCALE[name]]
            elif name == 'scripted_banks':
                value = len(self.scripted_pool())
            elif name == 'scripted_level_index':
                value = SCRIPTED_LEVELS.index(self.fixed.get('scripted_level', 'hard'))
            else:
                raise ConfigError(f'[params.{name}] needs a default (in [fixed] or default = ...)')
            _require(spec['min'] <= value <= spec['max'],
                     f'[params.{name}] default {value} is outside [{spec["min"]}, {spec["max"]}]')
            values[name] = value
        return values

    # ---- Flag rendering ------------------------------------------------------

    def is_integer(self, name):
        spec = self.params.get(name, {})
        return (bool(spec.get('integer')) or name in ALWAYS_INTEGER
                or spec.get('distribution') in ('int_uniform', 'uniform_pow2'))

    def trial_arguments(self, hypers, output, seed=None):
        """train.py argv (without the interpreter/script) for one trial.

        hypers: the suggestion dict ({param name: value}); values are rounded
        for integer parameters here, not in the space, so the optimizer always
        observes exactly what it suggested.
        """
        merged = dict(self.fixed)
        head_scale = _head_scale_triple(merged.get('entropy_head_scale', '1,1,1'))
        for name, value in hypers.items():
            if name in HEAD_SCALE:
                head_scale[HEAD_SCALE[name]] = max(0.0, min(100.0, float(value)))
            elif name == 'scripted_banks':
                pool = self.scripted_pool()
                merged['scripted_bots'] = pool[:max(0, min(len(pool), round(value)))]
            elif name == 'scripted_level_index':
                merged['scripted_level'] = SCRIPTED_LEVELS[max(0, min(2, round(value)))]
            elif name == 'steps' and self.fixed_cost:
                continue
            else:
                merged[name] = round(value) if self.is_integer(name) and not isinstance(value, int) else value
        merged['entropy_head_scale'] = ','.join(_number(s) for s in head_scale)
        if seed is not None:
            merged['seed'] = seed
        # train.py requires 0 < entropy_floor <= entropy; when entropy is swept
        # below the fixed floor, cap the floor at half the coefficient so the
        # trial still runs a decaying schedule instead of crashing.
        if 'entropy' in merged and 'entropy_floor' in merged:
            merged['entropy_floor'] = min(merged['entropy_floor'], merged['entropy'] / 2)
        arguments = ['--output', str(output)]
        for name in sorted(merged):
            value = merged[name]
            flag = '--' + name.replace('_', '-')
            if name in STORE_TRUE:
                if value:
                    arguments.append(flag)
                continue
            arguments += [flag, _render(value)]
        return arguments


def _head_scale_triple(value):
    parts = value if isinstance(value, list) else str(value).split(',')
    _require(len(parts) == 3, 'entropy_head_scale needs three components (aim,buttons,hold)')
    return [float(p) for p in parts]


def _number(value):
    if isinstance(value, int):
        return str(value)
    as_float = float(value)
    return str(int(as_float)) if as_float == int(as_float) and abs(as_float) < 1e15 else repr(as_float)


def _render(value):
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    if isinstance(value, float):
        _require(math.isfinite(value), f'Non-finite flag value {value}')
        return _number(value)
    return str(value)


def load(path):
    path = Path(path)
    with path.open('rb') as handle:
        raw = tomllib.load(handle)
    name = raw.get('sweep', {}).get('name', path.stem)
    _require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', name)),
             f'Sweep name {name!r} must be [A-Za-z0-9_-]+')
    return SweepSpec(raw, name)
