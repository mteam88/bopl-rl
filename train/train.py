"""Train native recurrent PPO against a PFSP self-play pool (league/pfsp.py)."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT / 'native/tools'), str(ROOT / 'train/sweep')]
from common import atomic_json
import counters
from evaluation_protocol import (SCRIPTED_KINDS, SCRIPTED_LEVELS, checkpoint_identity, opponents,
                                 DEFAULT_EVAL_SCRIPTED, DEFAULT_LOADOUT, map_ids, loadout_ids,
                                 loadout_pool, resolve_loadouts, ability_names)
from native_checkpoint import write_checkpoint
import layout
from native_metrics import empty_counters, update
import native_selfplay
from native_vector import NativeVector
import bank_kits
import reference
import scripted_roster
sys.path[:0] = [str(ROOT)]
from league import registry


def maps(value):
    try:
        return map_ids(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def map_mask_chunks(selected):
    """Four exact 31-bit integers consumed by binding.c as a 128-bit mask."""
    return [sum(1 << (map_id - 31*chunk) for map_id in selected
                if 31*chunk <= map_id < 31*(chunk+1)) for chunk in range(4)]


def _arg(parser, function):
    def parse(value):
        try:
            return function(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
    return parse


def install_loadout_weights(module, weights):
    """Per-ability pool weights ([[id, weight], ...], a JSON-safe pair list); empty keeps the uniform draw."""
    import ctypes
    storage=(ctypes.c_uint16*64).in_dll(ctypes.CDLL(module.__file__),'bopl_loadout_weights')
    weights=dict((int(ability), int(weight)) for ability, weight in weights)
    for ability in range(64):
        storage[ability]=weights.get(ability,0)


def loadout_weights(text):
    """'Grenade=4,Rock=4,7=2' -> {id: weight}; unnamed pool abilities weigh 1."""
    ids={name.lower(): ability for ability, name in ability_names().items()}
    weights={}
    for item in text.split(','):
        key, _, value = item.partition('=')
        key=key.strip().lower()
        ability=int(key) if key.isdigit() else ids.get(key)
        if ability is None or not value.strip().isdigit() or not 1 <= int(value) <= 1000:
            raise argparse.ArgumentTypeError(f'expected ability=weight (1..1000), got {item!r}')
        weights[ability]=int(value)
    return weights


def install_loadout_set(module, rows):
    """Copy parsed lineups into the C module; episode selection remains native."""
    import ctypes
    library=ctypes.CDLL(module.__file__)
    ctypes.c_int.in_dll(library,'bopl_loadout_set_count').value=len(rows)
    storage=(ctypes.c_uint8*(256*6)).in_dll(library,'bopl_loadout_sets')
    for index in range(len(storage)):
        storage[index]=0
    for row_index,row in enumerate(rows):
        for column,value in enumerate(row):
            storage[row_index*6+column]=value


class StyleBinding:
    """The compiled style program in the C environment (docs/STYLES.md part 3).

    scale() sets the fade before a rollout starts: 1 until fade.start of the
    run's learner steps, linear to 0 at fade.end, 0 after. read() turns the
    vector's cumulative counters (bopl_vec_style) into the rollout's per-term
    record: the unscaled payout, the reward actually paid (scale x payout),
    the decisions it fired on, per learner decision. Without --style both are
    no-ops and nothing is installed (the reward is bit-identical).
    """
    def __init__(self, module, program):
        import ctypes
        self.program, self.current, self.last = program, 0.0, None
        if program is None:
            return   # nothing installed: the binding's style pointer stays NULL
        lib = ctypes.CDLL(module.__file__)
        lib.bopl_style_install.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_char_p, ctypes.c_int]
        lib.bopl_style_install.restype = ctypes.c_int
        lib.bopl_style_set_scale.argtypes = [ctypes.c_float]
        lib.bopl_vec_style.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.bopl_vec_style.restype = ctypes.c_int
        self.lib, self.ctypes = lib, ctypes
        error = ctypes.create_string_buffer(512)
        if lib.bopl_style_install(program.data, len(program.data), error, len(error)):
            raise SystemExit(f'--style: the C loader refused the program: {error.value.decode()}')
        lib.bopl_style_set_scale(0.0)

    def scale(self, learner_steps, total):
        if self.program is None:
            return
        start, end = self.program.fade
        progress = learner_steps / max(1, total)
        value = 1.0 if progress <= start else 0.0 if progress >= end else (end - progress) / (end - start)
        self.current = value
        self.lib.bopl_style_set_scale(value)

    def read(self, vector):
        if self.program is None:
            return None
        terms = self.program.term_names
        buffer = (self.ctypes.c_double * (1 + 2 * len(terms)))()
        if self.lib.bopl_vec_style(self.ctypes.c_void_p(vector.gpu_obs_ptr), buffer, len(buffer)) != len(buffer):
            raise RuntimeError('bopl_vec_style: the vector or the installed program changed')
        now = list(buffer)
        last = self.last or [0.0] * len(now)
        self.last = now
        decisions = now[0] - last[0]
        per = lambda x: x / decisions if decisions else 0.0
        out = dict(scale=self.current, decisions=decisions, terms={})
        for i, name in enumerate(terms):
            pay, fires = now[1 + i] - last[1 + i], now[1 + len(terms) + i] - last[1 + len(terms) + i]
            out['terms'][name] = dict(pay=pay, paid=self.current * pay, fires=fires,
                                      pay_per_decision=per(pay), fire_rate=per(fires))
        out['paid_per_decision'] = per(sum(t['paid'] for t in out['terms'].values()))
        return out


def install_style(module, program):
    return StyleBinding(module, program)


def scripted_bots(value):
    """Comma-separated bot kinds (bopl_bots.h) that take over the last banks."""
    names = [x for x in value.split(',') if x]
    if any(name not in SCRIPTED_KINDS for name in names):
        raise argparse.ArgumentTypeError(f'Expected bot kinds from {sorted(SCRIPTED_KINDS)}')
    return names


def scripted_roster_arg(value):
    """Comma-separated kind-level names, or `default` for the 12 v2 kind-levels (search bots are opt-in)."""
    try:
        return scripted_roster.parse_roster(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def resolve_scripted(scripted_kinds, level, roster, banks=0):
    """Roster names and how many last banks they occupy.

    `--scripted-roster` is the list of (kind, level) entries. `--scripted-bots`
    without a roster is a one-entry-per-kind roster at `--scripted-level`. Bank
    count is `--scripted-banks` when given, else len(scripted-bots) when that
    flag is set, otherwise 1 if a roster exists. Every scripted bank draws from
    the same roster (bopl_puffer.h c_reset), so banks is a plain sweepable count.
    """
    names = list(roster or [])
    if not names and scripted_kinds:
        names = [f'{kind}-{level}' for kind in scripted_kinds]
    if banks:
        n_banks = banks if names else 0
    else:
        n_banks = len(scripted_kinds) if scripted_kinds else (1 if names else 0)
    return names, n_banks


def set_scripted_curriculum(_C, share, entries, weights):
    """Write share + roster kind/level/weight into the C environment (ctypes globals)."""
    import ctypes
    lib = ctypes.CDLL(_C.__file__)
    ctypes.c_float.in_dll(lib, 'bopl_scripted_share').value = float(share)
    n = len(entries)
    ctypes.c_int.in_dll(lib, 'bopl_roster_n').value = n
    kinds = (ctypes.c_uint8 * scripted_roster.ROSTER_MAX).in_dll(lib, 'bopl_roster_kind')
    levels = (ctypes.c_uint8 * scripted_roster.ROSTER_MAX).in_dll(lib, 'bopl_roster_level')
    warr = (ctypes.c_float * scripted_roster.ROSTER_MAX).in_dll(lib, 'bopl_roster_weight')
    for i, (entry, weight) in enumerate(zip(entries, weights)):
        kinds[i] = entry['kind']
        levels[i] = entry['level']
        warr[i] = float(weight)


def entropy_coefficient(learner_steps, initial, floor, half_life):
    """Exponential decay in learner decisions, independent of the run budget."""
    return max(floor, initial * math.exp2(-learner_steps / half_life))


def prio_beta(a, beta0, learner_steps):
    """The prioritized-replay beta for the next update; -1 keeps the native epoch anneal."""
    if a.prio_beta_const is not None:
        return a.prio_beta_const
    if a.prio_beta_fix:
        return beta0 + (1 - beta0) * min(1.0, learner_steps / a.steps)
    return -1.0


def schedule_fixes(a):
    """The opt-in schedule / encoder fixes, as config.json records them."""
    return dict(prio_beta_fix=a.prio_beta_fix, prio_beta_const=a.prio_beta_const, anneal_lr_fix=a.anneal_lr_fix,
                v4_full_rows=a.v4_full_rows)


DEFAULT_FIXES = dict(prio_beta_fix=0, prio_beta_const=None, anneal_lr_fix=0, v4_full_rows='0')


# Resume (--resume): a run directory is restarted from its last checkpoint and
# may be given a larger --steps (every schedule is a function of learner
# decisions or constant, never of the run budget). Restored: policy weights,
# the self-play pool (entries, per-bank opponents, PFSP draws, sampling RNG), the
# frozen banks' weights, the native step counter (pool snapshot names and
# intervals), and the cumulative environment counters, which the C vector
# restarts at zero and this process offsets, the optimizer state (Muon momentum,
# or Adam's moments and step count), and the
# --ema-decay weight average (when the decay is unchanged; else it restarts at
# the resumed weights).
# Not restored: the recurrent state, which starts as at an episode boundary,
# and the native trainer's epoch, so --anneal-lr's cosine and the native
# prioritized-replay beta restart at every resume unless --anneal-lr-fix /
# --prio-beta-fix drive them from the restored learner steps (schedule_fixes,
# which a resume may change; the change is recorded).
RESUME_HYPERS = ('workers', 'buffers', 'banks', 'seed', 'encoder', 'hidden', 'layers', 'horizon',
                 'learner_passes', 'learning_rate', 'anneal_lr', 'episode_ticks', 'stocks', 'training_maps', 'reward_kill',
                 'reward_timeout', 'reward_self_death', 'pool_size', 'pfsp_p', 'min_games', 'snapshot_interval',
                 'opponent_timeout', 'pool_seed', 'head_scale', 'entropy_free_rows',
                 'scripted_bots', 'scripted_level', 'scripted_roster', 'scripted_share_min',
                 'scripted_share_max', 'n_scripted', 'aim_fourier', 'aim_ar', 'v4_shape', 'loadouts',
                 'historical_every', 'team_every', 'style_sha256', 'optimizer', 'weight_decay', 'adam_lr')
# The entropy schedule is a pure function of these and the learner step,
# recomputed every update, so a resume may change it; the change is recorded.
RESUME_SCHEDULE = ('entropy', 'entropy_floor', 'entropy_half_life')
# --optimizer muon-adamw: the v4 tensors AdamW updates (embedding and output heads).
ADAMW_TENSORS = ('embedding', 'actor', 'critic_out')
# Policy v4 (train/bopl_v4.cu) network shape: flag name -> (env var the
# CUDA module reads, layout.V4_DEFAULTS key). Recorded in config.json / the checkpoint sidecar.
V4_SHAPE = dict(set_dim=('BOPL4_SET_DIM', 'set_dim'), set_inner=('BOPL4_SET_INNER', 'set_inner'),
                flat_dim=('BOPL4_FLAT_DIM', 'flat_dim'), ffn_dim=('BOPL4_FFN_DIM', 'ffn_dim'),
                critic_dim=('BOPL4_CRITIC_DIM', 'critic_dim'), critic_privileged=('BOPL4_CRITIC_PRIVILEGED', 'critic_privileged'),
                critic_detach=('BOPL4_CRITIC_DETACH', 'critic_detach'), mean_pool=('BOPL4_MEAN_POOL', 'mean_pool'),
                ablate=('BOPL4_ABLATE', 'ablate'))   # writer bitmask: 1 relational, 2 terrain + rays, 4 rays16 (ablations), 8 layout v4.1 aim / line-of-sight rays, 16 layout v4.2 observation fixes + Missile/Spike kill credit
COUNTER_KEYS = ('learner_steps', 'game_steps', 'matches', 'opponent_game_decisions', 'kills')
# Heads the environment masks while a latch is in force (bopl_write_mask):
# buttons and hold. Aim is live every decision.
LATCHED_HEADS = (1, 2)
FREE_FRACTION_FLOOR = 1 / layout.HOLD_LENGTHS[-1]


def free_fraction(env_log):
    """Fraction of the rollout's decisions that were free (not latched), from the env's hold stats.

    None until completed episodes report; the per-episode means share one denominator, so
    their ratio is the fraction. The floor is what the longest latch alone would give.
    """
    free = sum(env_log.get(f'hold_{n}', 0.0) for n in layout.HOLD_LENGTHS)
    latched = env_log.get('latched_decisions', 0.0)
    if free + latched <= 0:
        return None
    return max(FREE_FRACTION_FLOOR, min(1.0, free / (free + latched)))


def recorded_hypers(config):
    """The RESUME_HYPERS of a run as config.json recorded them."""
    pool, train, env = config['selfplay'], config['native']['train'], config['native']['env']
    return dict(workers=config['workers'], buffers=config['native']['vec']['num_buffers'], banks=env['banks'],
                seed=config['seed'], encoder=config['encoder'], hidden=config['hidden'], layers=config['layers'],
                horizon=train['horizon'], learner_passes=config['learner_passes'], learning_rate=train['learning_rate'],
                anneal_lr=train['anneal_lr'], entropy=config['entropy'], entropy_floor=config['entropy_floor'],
                entropy_half_life=config['entropy_half_life'], episode_ticks=config['episode_ticks'], stocks=config['stocks'],
                training_maps=config['training_maps'], reward_kill=config['reward']['kill'],
                reward_timeout=config['reward']['timeout'],
                reward_self_death=-config['reward'].get('self_death', 0.0), pool_size=pool['max_size'], pfsp_p=pool['pfsp_p'],
                min_games=pool['min_games'], snapshot_interval=pool['snapshot_interval'],
                opponent_timeout=pool['opponent_timeout'],
                pool_seed=pool['seed'], head_scale=config['entropy_head_scale'],
                entropy_free_rows=config.get('entropy_free_rows', 0),
                scripted_bots=config.get('scripted_bots', []), scripted_level=config.get('scripted_level', 'hard'),
                scripted_roster=config.get('scripted_roster', []),
                scripted_share_min=config.get('scripted_share_min', 0.0),
                scripted_share_max=config.get('scripted_share_max', 1.0),
                n_scripted=config.get('scripted_banks', len(config.get('scripted_bots', [])) or (1 if config.get('scripted_roster') else 0)),
                aim_fourier=config.get('aim_fourier', 0), aim_ar=config.get('aim_ar', 0),
                v4_shape=config.get('v4_shape'), loadouts=config.get('loadouts',resolve_loadouts()),
                historical_every=env.get('historical_every', 2), team_every=env.get('team_every', 0),
                style_sha256=(config.get('style') or {}).get('sha256'), optimizer=config.get('optimizer', 'muon'),
                weight_decay=config.get('weight_decay', 0.), adam_lr=config.get('adam_lr'))


def offset_counters(counts, base):
    """Cumulative counters of this process plus those of the runs before it."""
    if base is None:
        return counts
    out = dict(counts)
    for key in COUNTER_KEYS:
        out[key] = counts[key] + base[key]
    out['buttons'] = [x + y for x, y in zip(counts['buttons'], base['buttons'])]
    out['telemetry'] = [x + y for x, y in zip(counts['telemetry'], base['telemetry'])]
    for key in ('ability_activations','entity_rows','entity_overflow'):
        out[key] = [x + y for x, y in zip(counts[key], base.get(key,[0]*len(counts[key])))]
    out['map_matches'] = {m: counts['map_matches'].get(m, 0) + base['map_matches'].get(m, 0)
                          for m in set(counts['map_matches']) | set(base['map_matches'])}
    return out


def pool_state_json(pool, ledger, *, counts, elapsed, global_step, update_index, checkpoint, momentum, ema=None):
    return dict(schema=1, pool={k: v for k, v in pool.items() if k != 'rng'},
                rng=pool['rng'].bit_generator.state, ledger=ledger, counts=counts, elapsed=elapsed,
                global_step=int(global_step), update_index=update_index, checkpoint=str(checkpoint),
                momentum=str(momentum), **(dict(ema=str(ema)) if ema else {}))


def load_pool_state(record):
    import numpy as np
    pool = dict(record['pool'])
    pool['rng'] = np.random.default_rng()
    pool['rng'].bit_generator.state = record['rng']
    return pool, record['ledger']


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true',
                   help='Continue the run in --output from its last checkpoint (same hyperparameters; --steps may grow)')
    p.add_argument('--init-weights', type=Path,
                   help='Start a fresh run from this checkpoint instead of random init (a league main '
                        'continued against a new population). Its architecture must match the run\'s '
                        'exactly; step counters, schedules, the optimiser and the pool start fresh')
    p.add_argument('--kl-reference', type=Path,
                   help='KL-to-reference (AlphaStar-style anchor): a frozen checkpoint of the run\'s architecture '
                        'run on the learner rows; each learner decision pays --kl-coef * (log pi(a) - log ref(a)) '
                        'in the reward, the sampled KL(pi || ref). Uses the per-bank rollout chain (slower). '
                        'Logged as kl_reference (mean sampled KL per decision)')
    p.add_argument('--kl-coef', type=float, default=0.0,
                   help='Reward charged per nat of sampled KL to --kl-reference per decision (0 = measure only)')
    p.add_argument('--game', type=Path)
    p.add_argument('--steps', type=int, default=100_000, help='Actual learner decisions (excludes frozen players)')
    p.add_argument('--workers', type=int, default=16, help='Number of simultaneous two-player games')
    p.add_argument('--buffers', type=int, choices=(1, 2, 4, 8),
                   help='Rollout buffers (independent buffer threads + CUDA streams over one env pool). 4 measured +11%% learner/s over 2 '
                        'at 128 workers on the f64 sim (2026-09-10): GPU sync of one buffer overlaps env stepping of the others')
    p.add_argument('--banks', type=int, choices=range(2, 9), default=4)
    p.add_argument('--historical-every', type=int, choices=(1, 2), default=2,
                   help='every k-th game is learner vs a frozen bank, the rest learner vs learner. 2: half the games, '
                        '2/3 of the learner rows self-play (a main agent). 1: every game against the banks, no '
                        'self-play rows (an exploiter training only against its frozen target(s))')
    p.add_argument('--team-every', type=int, default=0,
                   help='team play (docs/TEAM_PLAY.md): the self-play game of every k-th self-play/historical group is a '
                        '2v2 team game of four learner players (0 = none; needs --historical-every 2). 2 puts half the '
                        'learner rows in team games')
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--encoder', choices=('linear', 'set', 'v4'), default='linear',
                   help='v4: observation layout 4 and the policy v4 encoder/decoder (train/bopl_v4.cu)')
    for name, (_, key) in V4_SHAPE.items():
        p.add_argument('--v4-' + name.replace('_', '-'), type=int, default=layout.V4_DEFAULTS[key], dest='v4_' + name,
                       help=f'policy v4 shape (encoder v4 only; default {layout.V4_DEFAULTS[key]})')
    p.add_argument('--dump-rows', type=Path, default=None, metavar='NPZ',
                   help='after the first rollout, save the observations and the trainer\'s own decoder rows for '
                        '64 learner rows of buffer 0 at step 0 (zero recurrent state; _C.bopl_policy_rows) next to '
                        'the initial checkpoint and stop: the fixture train/check_dump.py holds the CPU evaluator '
                        'and the torch reference to. Also saves the batched rollout chain\'s rows on the same '
                        'observations (_C.bopl_chain_rows)')
    p.add_argument('--chain-reference', type=Path, default=None, metavar='NPZ',
                   help='with --dump-rows: a dump from another build; run this build\'s rollout chain on its '
                        'observations with its checkpoint loaded and save the rows as chain_rows_reference, for '
                        'check_dump.py to compare bit for bit')
    p.add_argument('--v4-row-budget', default=None, metavar='P,S,E,L',
                   help='policy v4: packed rows per sample the set MLPs hold for players, slots, entities, slimes '
                        '(bopl_v4.cu DEFAULT_BUDGET 6,12,20,8); rows past the budget are dropped and counted (set_overflow)')
    p.add_argument('--v4-full-rows', default='0', metavar='0|1|P,S,E,L',
                   help='policy v4: 1 sizes the training set buffers for every row, so no row is dropped (training '
                        'sees the rollout\'s rows and no sample depends on its minibatch); the budget only sizes the '
                        'GEMM buckets and set_overflow counts the rows kept past it. P,S,E,L keeps each sample\'s '
                        'first rows of each set (rows past a cap are cut from that sample only, counted in set_cut): '
                        'less memory, exact while no sample exceeds a cap (6,12,48,8 covers 2v2). '
                        '0 (default) drops past the budget in batch order')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--learner-passes', type=int, default=1, help='Native minibatches per rollout; each has one learner batch of samples')
    p.add_argument('--minibatch-div', type=int, default=1,
                   help='Split each learner batch into this many minibatches (optimizer steps); must divide the learner rows. '
                        'Throughput knob: smaller minibatches give the 16-SM training partition less work per GEMM but more launches; default 1')
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--optimizer', choices=('muon', 'adam', 'muon-adamw'), default='muon',
                   help='muon (default); adam: torch-style AdamW on every parameter with the config beta1/beta2/eps '
                        '(0.95/0.999/1e-12) and the same gradient clip; muon-adamw: Muon, but AdamW at --adam-lr on '
                        f'the v4 {"/".join(ADAMW_TENSORS)} tensors and any 1-D one. AdamW state resumes with --resume')
    p.add_argument('--weight-decay', type=float, default=0.,
                   help='Decoupled weight decay for either optimizer, w *= 1 - lr*wd every optimizer step '
                        '(Moonlight/AdamW style, follows the LR schedule; under muon-adamw every tensor decays at the '
                        'Muon LR); 0 (default) = off')
    p.add_argument('--adam-lr', type=float, default=None,
                   help='muon-adamw: peak learning rate of the AdamW tensors (scaled with the LR schedule); '
                        'default --learning-rate')
    p.add_argument('--anneal-lr', type=int, choices=(0, 1), default=0,
                   help='0 keeps learning rate constant; 1 uses native cosine decay over the run budget')
    p.add_argument('--min-lr-ratio', type=float, default=0.,
                   help='Final/base learning-rate ratio when --anneal-lr 1')
    p.add_argument('--anneal-lr-fix', type=int, choices=(0, 1), default=0,
                   help='1: the --anneal-lr cosine follows learner steps / --steps (restored on --resume); 0 (default) '
                        "keeps the native per-process epoch, which restarts the cosine at every --resume")
    p.add_argument('--prio-beta-fix', type=int, choices=(0, 1), default=0,
                   help='1: prioritized-replay beta anneals beta0 -> 1 over learner steps / --steps (restored on --resume). '
                        '0 (default) keeps the native beta0 + (1 - beta0) alpha epoch / total_epochs: the epoch '
                        'restarts at every --resume and the alpha factor stops it at 0.84')
    p.add_argument('--prio-beta-const', type=float, default=None,
                   help='Fixed prioritized-replay beta in [0, 1] (factorial arms); excludes --prio-beta-fix')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='Polyak average of the weights, ema = d*ema + (1-d)*w after every optimizer step, saved as '
                        'checkpoints/policy-<step>-ema.bin beside each checkpoint; 0 (default) = off. Half-life '
                        'ln(2)/(1-d) optimizer steps (0.99997: 23k)')
    p.add_argument('--gae-lambda', type=float, default=None,
                   help='GAE lambda; default keeps the PufferLib config value (0.90)')
    p.add_argument('--gamma', type=float, default=None,
                   help='Discount per decision; default keeps the PufferLib config value (train/bopl.ini, 0.995)')
    p.add_argument('--prio-alpha', type=float, default=None,
                   help='Prioritized replay exponent on |advantage| (0 = uniform sampling); default keeps the '
                        'PufferLib config value (0.8)')
    p.add_argument('--entropy', type=float, default=.01)
    p.add_argument('--entropy-floor', type=float, default=.0005)
    p.add_argument('--entropy-half-life', type=int, default=30_000_000,
                   help='Learner decisions per halving of the entropy coefficient, until its floor')
    p.add_argument('--entropy-head-scale', default='1,1,1',
                   help='Per-head multipliers on the entropy bonus (aim,buttons,hold); logged entropies stay unscaled')
    p.add_argument('--entropy-free-rows', type=int, choices=(0, 1), default=1,
                   help='1 normalises the button/hold entropy bonus over free (unlatched) rows: the head scale is '
                        'divided by the previous rollout\'s free-decision fraction (latched rows carry no entropy)')
    p.add_argument('--aim-fourier', type=int, default=0, choices=range(0, layout.AIM_MAX_HARMONICS + 1), metavar='K',
                   help='Aim head as a Fourier series over angle: the decoder emits a neutral logit and K cos/sin '
                        'harmonics per aim block instead of 256 bins (train/bopl_aim.h); 0 keeps raw bins')
    p.add_argument('--aim-ar', type=int, choices=(0, 1), default=0,
                   help='1 conditions the aim head on the sampled button combo: one aim block per combo, buttons '
                        'sampled before aim (in the trainer, the CPU evaluator and the in-game agent)')
    p.add_argument('--episode-ticks', type=int, default=7200)
    p.add_argument('--reward-kill', type=float, default=0.,
                   help='Per-kill bonus, also paid as a per-death penalty (game credit incl. self-kills)')
    p.add_argument('--reward-timeout', type=float, default=-.5,
                   help='Terminal reward for both players when the episode hits --episode-ticks (<= 0)')
    p.add_argument('--reward-self-death', type=float, default=0.,
                   help='Extra penalty (>= 0) per death the game files as self-inflicted or environmental '
                        '(own weapon, water, falling), on top of --reward-kill\'s death penalty')
    p.add_argument('--stocks', type=int, default=0, help='Smash-style extra-life setting; 0 = vanilla Bopl, one life per round')
    p.add_argument('--training-maps', type=maps, default=[0],
                   help='identity map IDs, inclusive ranges, all, or named sets from native/data/map-sets.json')
    p.add_argument('--loadout', type=_arg(p,loadout_ids), default=list(DEFAULT_LOADOUT),
                   help='fixed three-slot loadout (default: 2,4,14)')
    p.add_argument('--loadout-pool', type=_arg(p,loadout_pool),
                   help='native per-episode slot pool: IDs, all, or all-<excluded IDs>')
    p.add_argument('--loadout-set', type=Path,
                   help='lineup file: three shared or six player-ordered ability IDs per line')
    p.add_argument('--loadout-no-dup', action='store_true',
                   help='sample distinct slots within each player (duplicates are legal by default)')
    p.add_argument('--loadout-weights', type=loadout_weights, default={},
                   help="pool draw weights, e.g. 'Grenade=4,Rock=4' (ability names or IDs); others weigh 1")
    p.add_argument('--loadout-learner-first', action='store_true',
                   help='with --loadout-set (six-id rows) and frozen banks: each row is [learner kit | opponent kit] '
                        'whichever seat the learner draws, so an exploiter always plays its own kit against its '
                        "target's (league exploiters with a kit of their own, e.g. a camper)")
    p.add_argument('--mirror-share', type=float, default=0.,
                   help='probability both players receive player 0\'s loadout')
    p.add_argument('--checkpoint-every', type=int, default=1_000_000, help='Learner decisions between immutable snapshots')
    p.add_argument('--metrics-every', type=int, default=0,
                   help='Learner decisions between metrics.jsonl rows and console lines (0 = every update; '
                        'the last update is always written). Each row is one update\'s record, not an average '
                        'over the interval. A row per update reached 70 GB over 100B steps')
    p.add_argument('--pool-size', type=int, default=16, help='Historical snapshots kept by the self-play pool')
    p.add_argument('--pfsp-p', type=float, default=1.,
                   help='PFSP hardness exponent: opponent weight is (1-our score)^p. 0 is uniform, '
                        'larger concentrates harder on opponents that beat us (league/pfsp.py)')
    p.add_argument('--pfsp-f', choices=('hard', 'var'), default='hard',
                   help='PFSP weighting: hard is (1-x)^p (--pfsp-p), var is x(1-x), peaking at even matchups '
                        '(--pfsp-p unused). May change on --resume')
    p.add_argument('--min-games', type=int, default=256, help='Completed games per bank before the band gate can retire the opponent')
    p.add_argument('--snapshot-interval', type=int, default=10_000_000, help='Learner decisions between pool snapshots taken regardless of win rate')
    p.add_argument('--opponent-timeout', type=int, default=50_000_000, help='Learner decisions a bank keeps one opponent before a forced swap')
    p.add_argument('--pool-seed', type=int, help='Opponent sampling seed (defaults to --seed)')
    p.add_argument('--opponents', default='league',
                   help="League members this run plays besides its own snapshots: 'league' for "
                        "every member with weights, 'self' for none, or a comma list of member "
                        "names (how an exploiter aims at one member). PFSP draws over all of them "
                        "together on the same statistic, so there is no separate share to set. "
                        "On --resume the pool's league members become exactly this set: new "
                        "members join, unnamed ones stop being drawn (how a league loop feeds a "
                        "continuing run).")
    p.add_argument('--bank-groups', default='',
                   help="Comma list, one per policy bank, of the candidates each bank draws from: "
                        "all (own snapshots and league members), league (league members only: an "
                        "exploiter), exploiters (league exploiters only: AlphaStar's main-agent "
                        "branch), pinned (--pinned-opponents only). Empty is all everywhere. See "
                        "native_selfplay.groups. May change on --resume")
    p.add_argument('--pinned-opponents', default='',
                   help='Comma list of league members that play their own declared kits (league.json '
                        '`loadouts`) from the frozen seat (docs/LEAGUE.md "Kit pinning"), or `auto`: '
                        'every --opponents member that declares kits. The learner keeps its own kit draw '
                        'in those games. With a bank of group `pinned` only such banks draw them (their '
                        'share of the games is those banks\' share); without one they are ordinary '
                        'candidates of every group, pinned wherever drawn. May change on --resume')
    p.add_argument('--scripted-bots', type=scripted_bots, default='',
                   help='Comma-separated bot kinds that take over the last banks. Without '
                        '--scripted-roster each kind is a one-entry roster at --scripted-level; '
                        'with a roster this only sets how many banks draw from it')
    p.add_argument('--scripted-level', choices=sorted(SCRIPTED_LEVELS, key=SCRIPTED_LEVELS.get), default='hard')
    p.add_argument('--scripted-roster', nargs='?', const='default', default=None, type=scripted_roster_arg,
                   help='Comma-separated kind-level opponents the scripted bank matchmakes among '
                        f'(default roster: {",".join(scripted_roster.DEFAULT_ROSTER)}). '
                        'Pass the flag with no value for that default. Empty / omitted with no '
                        '--scripted-bots disables scripted banks')
    p.add_argument('--scripted-banks', type=int, default=0,
                   help='How many of the last banks draw from the scripted roster (0 = derive: '
                        'len(--scripted-bots), else 1 with a roster). Must leave one policy bank')
    p.add_argument('--scripted-share-max', type=float, default=1.0,
                   help='Upper clamp on the scripted bank\'s bot-match share (1 = the bank always '
                        'plays a roster bot while some entry is a coin flip)')
    p.add_argument('--scripted-share-min', type=float, default=0.0,
                   help='Lower clamp on the scripted bank\'s bot-match share (0 = drop the bot '
                        'once every roster entry is beaten or hopeless)')
    p.add_argument('--scripted-elo', default='',
                   help="Comma list name=elo: the starting roster Elo of scripted entries the league has "
                        "not rated yet (e.g. rusher-hard=450). Rated entries keep their league rating")
    p.add_argument('--scripted-opponents', default=DEFAULT_EVAL_SCRIPTED,
                   help='Comma-separated kind-level scripted opponents for the periodic evaluations (evaluate.py); empty disables')
    p.add_argument('--cudagraphs', type=int, default=1, help='Warmup updates before capture; -1 disables capture; 0 is unsupported')
    p.add_argument('--no-eval', action='store_true', help='Disable diagnostic evaluation for isolated throughput measurements')
    p.add_argument('--eval-every', type=int, default=1_000_000, help='Learner decisions between diagnostic evaluations')
    p.add_argument('--evaluation-maps', type=maps)
    p.add_argument('--heldout-maps', type=maps)
    p.add_argument('--evaluation-baseline', type=Path)
    p.add_argument('--evaluation-opponents', type=Path)
    p.add_argument('--evaluation-games', type=int, default=8)
    p.add_argument('--heldout-games', type=int, default=4)
    p.add_argument('--frozen-games', type=int, default=16)
    p.add_argument('--evaluation-workers', type=int, default=2)
    p.add_argument('--no-in-run-rating', action='store_true',
                   help='do not rate every evaluation checkpoint against the league in-run (league candidate, '
                        'nice 19, 2 workers, CPU only, non-blocking; skipped while the previous one runs)')
    p.add_argument('--in-run-games', type=int, default=160, help='games of each in-run league rating')
    p.add_argument('--style', nargs='+', default=[], metavar='TOML',
                   help='style specs (docs/STYLES.md, styles/*.toml; several or comma-separated) compiled into '
                        'one program whose bonus every env adds to the learner reward, faded by the first '
                        "spec's fade over the run's learner steps")
    p.add_argument('--style-fade-steps', type=int,
                   help="learner steps the style fade is measured over (default --steps); the league loop sets "
                        "the exploiter's timeout so the fade stays monotonic as --resume grows --steps")
    a = p.parse_args()
    a.style = [Path(x) for part in a.style for x in part.split(',') if x]
    a.style_program = a.style_sha256 = None
    if a.style:
        sys.path.insert(0, str(ROOT/'train'))
        import style_compile
        try:
            a.style_program = style_compile.compile_files(a.style)
        except (style_compile.CompileError, OSError) as error:
            p.error(f'--style: {error}')
        a.style_sha256 = a.style_program.sha256
    for name in ('steps', 'workers', 'horizon', 'layers', 'learner_passes', 'episode_ticks',
                 'checkpoint_every', 'eval_every', 'evaluation_workers', 'entropy_half_life',
                 'pool_size', 'min_games', 'snapshot_interval', 'opponent_timeout'):
        if getattr(a, name) < 1:
            p.error(f'{name.replace("_", "-")} must be positive')
    if a.metrics_every < 0:
        p.error('metrics-every must be non-negative')
    if not math.isfinite(a.pfsp_p) or a.pfsp_p < 0:
        p.error('Expected a finite, non-negative pfsp-p')
    if not math.isfinite(a.reward_self_death) or a.reward_self_death < 0:
        p.error('Expected a finite, non-negative reward-self-death')
    a.pool_seed = a.seed if a.pool_seed is None else a.pool_seed
    a.buffers = a.buffers or (2 if a.workers % 4 == 0 else 1)
    if a.workers % (a.historical_every * a.buffers):
        p.error('workers must be divisible by historical-every times the buffer count')
    if (a.workers // a.historical_every) % a.banks:
        p.error('the historical games (workers / historical-every) must divide evenly among the banks')
    if a.team_every and (a.team_every < 0 or a.historical_every != 2 or (a.workers // (2 * a.buffers)) % a.team_every):
        p.error('team-every needs historical-every 2 and must divide the groups of a buffer (workers / 2 / buffers)')
    a.scripted_bots = list(a.scripted_bots or [])
    a.scripted_roster = a.scripted_roster or []
    if a.scripted_banks < 0:
        p.error('scripted-banks must be >= 0')
    roster_names, n_scripted = resolve_scripted(a.scripted_bots, a.scripted_level, a.scripted_roster, a.scripted_banks)
    a.scripted_roster = roster_names
    a.n_scripted = n_scripted
    if n_scripted > a.banks:
        p.error(f'scripted-banks {n_scripted} exceeds --banks {a.banks}')
    if n_scripted and not roster_names:
        p.error('scripted-roster/scripted-bots produced no roster entries')
    if not (math.isfinite(a.scripted_share_min) and math.isfinite(a.scripted_share_max)
            and 0 <= a.scripted_share_min <= 1 and 0 <= a.scripted_share_max <= 1):
        p.error('scripted-share-min/max must be finite in [0, 1]')
    if a.scripted_share_min > a.scripted_share_max:
        # A sweep may propose min > max; that is a constant share at max, not a
        # rejected trial (the optimizer observes exactly what it suggested).
        print(f'scripted-share-min {a.scripted_share_min} > max {a.scripted_share_max}: clamping min to max', flush=True)
        a.scripted_share_min = a.scripted_share_max
    if a.hidden < 8 or a.hidden % 8:
        p.error('hidden must be a positive multiple of eight')
    if a.cudagraphs != -1 and a.cudagraphs < 1:
        p.error('cudagraphs must be -1 or at least 1')
    if not (math.isfinite(a.learning_rate) and a.learning_rate > 0
            and 0 < a.entropy_floor <= a.entropy and math.isfinite(a.entropy)):
        p.error('Expected positive learning rate and 0 < entropy-floor <= entropy')
    if not math.isfinite(a.min_lr_ratio) or not 0 <= a.min_lr_ratio <= 1:
        p.error('min-lr-ratio must be finite and within 0..1')
    if not 0 <= a.ema_decay < 1:
        p.error('ema-decay must be within [0, 1)')
    if a.prio_beta_const is not None and (a.prio_beta_fix or not 0 <= a.prio_beta_const <= 1):
        p.error('prio-beta-const must be within [0, 1] and excludes --prio-beta-fix')
    caps = a.v4_full_rows.split(',')
    if not (a.v4_full_rows in ('0', '1') or (len(caps) == 4 and all(c.isdigit() and int(c) > 0 for c in caps))):
        p.error('v4-full-rows must be 0, 1 or four positive row caps P,S,E,L')
    if not (math.isfinite(a.weight_decay) and a.weight_decay >= 0):
        p.error('weight-decay must be finite and >= 0')
    if a.optimizer == 'muon-adamw':
        if a.encoder != 'v4':
            p.error('--optimizer muon-adamw names v4 tensors; it needs --encoder v4')
        a.adam_lr = a.adam_lr or a.learning_rate
        if not a.adam_lr > 0:
            p.error('adam-lr must be positive')
    elif a.adam_lr is not None:
        p.error('--adam-lr only applies to --optimizer muon-adamw')
    try:
        head_scale = [float(part) for part in a.entropy_head_scale.split(',')]
    except ValueError:
        head_scale = []
    if len(head_scale) != layout.NUM_ATNS or not all(math.isfinite(s) and 0 <= s <= 100 for s in head_scale):
        p.error(f'entropy-head-scale must be {layout.NUM_ATNS} finite values in 0..100')
    a.head_scale = head_scale
    a.evaluation_maps = a.evaluation_maps or a.training_maps
    a.heldout_maps = a.heldout_maps or [m for m in range(counters.MAPS) if m not in a.training_maps][:2]
    if not a.no_eval:
        if not a.heldout_maps or set(a.heldout_maps) & set(a.training_maps + a.evaluation_maps):
            p.error('heldout maps must be disjoint from training and evaluation maps')
        if any(n < 2 or n % 2 for n in (a.evaluation_games, a.heldout_games, a.frozen_games)):
            p.error('evaluation game counts must be positive and even')
    try:
        a.loadouts=resolve_loadouts(a.loadout,a.loadout_pool,a.loadout_set,a.loadout_no_dup,a.mirror_share)
    except (ValueError,OSError) as error:
        p.error(str(error))
    if a.loadout_learner_first:
        if a.loadouts['mode'] != 'set' or a.loadouts['mirror_share'] or a.team_every:
            p.error('--loadout-learner-first needs --loadout-set, no --mirror-share and no --team-every')
        a.loadouts=dict(a.loadouts, learner_first=True)
    if a.loadout_weights:
        if a.loadouts['mode'] != 'pool' or set(a.loadout_weights) - set(a.loadouts['pool']):
            p.error('--loadout-weights needs --loadout-pool containing every weighted ability')
        a.loadouts=dict(a.loadouts, weights=[[ability, a.loadout_weights.get(ability, 1)] for ability in a.loadouts['pool']])
    a.pinned_opponents = [s for s in a.pinned_opponents.split(',') if s]
    if len(set(a.pinned_opponents)) != len(a.pinned_opponents):
        p.error('--pinned-opponents names a member twice')
    if 'auto' in a.pinned_opponents and a.pinned_opponents != ['auto']:
        p.error('--pinned-opponents auto stands alone')
    try:
        a.scripted_elo = {name: float(elo) for name, _, elo in
                          (item.partition('=') for item in a.scripted_elo.split(',') if item)}
    except ValueError:
        p.error('--scripted-elo expects name=elo pairs')
    # Kit pinning (bank_kits.py) is on with pinned members, or when a scripted
    # bank cannot play this run's kits or maps (below).
    a.bank_kits = bool(a.pinned_opponents)
    a.bot_kit = None
    if a.n_scripted and not all(scripted_roster.split_name(name)[0] == 'uniform' for name in roster_names):
        # A scripted bot only plays the kit and the maps it was written for, the ones
        # its league member declares (registry.kits, registry.plays_map); anywhere
        # else it presses whatever sits in the slot it wants. The uniform kind plays
        # any kit on any map: it presses nothing on purpose. With kit pinning the bot
        # plays its own kit whatever the run deals and leaves space maps to the bank's
        # frozen weights, so any run can face it.
        members = registry.by_name(registry.load(verify=False))
        bot_kits = {tuple(map(tuple, registry.kits(members[name]) or [])) for name in roster_names}
        if len(bot_kits) != 1 or len(next(iter(bot_kits))) != 1:
            p.error(f'scripted roster entries must share one kit to be dealt it, got {bot_kits}')
        bot = members[roster_names[0]]
        dealt = (registry.trained_loadouts(dict(loadouts=a.loadouts)) if a.loadouts['mode'] != 'pool'
                 else dict(mode='pool'))
        if (dealt['mode'] == 'pool' or registry.kits(dict(loadouts=dealt)) != registry.kits(bot)
                or not all(registry.plays_map(bot, m) for m in a.training_maps)):
            a.bank_kits = True
        if a.bank_kits:
            a.bot_kit = list(registry.kits(bot)[0])
    return a


def main():
    a = arguments()
    output = a.output.resolve()
    frozen = opponents(a.evaluation_opponents)
    baseline_identity = checkpoint_identity(a.evaluation_baseline) if a.evaluation_baseline else None
    # BOPL_RL_BACKEND=rust rolls out on the Rust sim (native/src/rollout,
    # built by native/tools/pgo_build.sh); anything else keeps the .NET/Unity
    # game resolution. The Rust backend needs no spare world (resets are
    # microseconds) and hosts every world in one runtime.
    # An explicit --game names a staged build of the sim (a map-catalogue
    # round the live queue's library does not carry yet); the default is the
    # tree's own build.
    if os.environ.get('BOPL_RL_BACKEND', '').lower() == 'rust':
        game = (a.game or ROOT/'native/target/release/libbopl_sim.so').resolve()
        if not game.exists():
            raise SystemExit(f'BOPL_RL_BACKEND=rust: {game} is missing; run native/tools/pgo_build.sh')
        os.environ.setdefault('BOPL_RL_INPROC_SPARE', '0')
        os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', str(1 << 20))
    else:
        game = reference.resolve_game(a.game)
    # The NativeAOT sim is one self-contained binary: only the
    # player_binary entry exists for it, the Unity-specific files do not.
    runtime = {key: hashlib.sha256(path.read_bytes()).hexdigest()
               for key, path in reference.player_sources(game).items() if path.exists()}
    a.v4_shape = {key: getattr(a, 'v4_' + name) for name, (_, key) in V4_SHAPE.items()} if a.encoder == 'v4' else None
    resume = None
    if a.resume:
        resume = json.loads((output/'pool/state.json').read_text())
        previous_config = json.loads((output/'config.json').read_text())
        recorded = recorded_hypers(previous_config)
        for name in RESUME_HYPERS:
            if recorded[name] != getattr(a, name):
                raise SystemExit(f'--resume: {name} was {recorded[name]!r}, now {getattr(a, name)!r}; only --steps '
                                 f'and the entropy schedule may change')
        schedule_changes = {name: [recorded[name], getattr(a, name)] for name in RESUME_SCHEDULE
                            if recorded[name] != getattr(a, name)}
        if a.steps < resume['counts']['learner_steps']:
            raise SystemExit(f"--resume: --steps {a.steps} is below the {resume['counts']['learner_steps']} already trained")
        if not Path(resume['checkpoint']).exists():
            raise SystemExit(f"--resume: checkpoint {resume['checkpoint']} is missing")
        (output/'game-logs').mkdir(exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    frozen_manifest = output/'evaluation-opponents.json' if frozen else None
    if frozen_manifest and not resume:
        atomic_json(frozen_manifest, dict(schema=2, opponents=frozen))
    # One GC heap under every world costs 25-45%
    # of throughput (each gen0 collection suspends every world); four worlds
    # per runtime measured within 5-15% of one runtime per world at a
    # fraction of the memory. Honour an explicit setting.
    os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', '4')
    # The rollout steps games on vecenv.h's own env pool, not
    # OpenMP: BOPL_ENV_THREADS stepping threads in total (buffer threads
    # included; default CPUs - 1, i.e. 11 on the 12-thread box: at w256 b4 the
    # rollout is CPU-bound on c_step x threads and 11 beat 9 by 4% (2026-09-19);
    # 12+ oversubscribes and stalls on preempted workers), BOPL_ENV_LPT=1
    # slowest-first claim order, BOPL_ENV_STEAL=1
    # buffer threads step other buffers' games while their own forward runs,
    # BOPL_ENV_SPIN_US=10 idle spin before a futex park, BOPL_GPU_WAIT=poll|sync,
    # BOPL_ENV_PIN=1 to pin threads, BOPL_ENV_PROF=1|2 pool statistics and the
    # c_step phase attribution on stderr.
    # BOPL_ASYNC=1 (default) overlaps each rollout with the previous batch's
    # training (see the update loop); 0 is the serial trainer. With overlap
    # BOPL_GPU_SPLIT=<SMs for inference>|auto|0 partitions the GPU between the
    # rollout and train streams (green contexts; auto = one half, 24/24 on
    # the 48-SM 5070; 0 shares the SMs on stream priority alone) and
    # BOPL_TRAIN_WAIT=block|spin picks how the trainer waits for its stream.
    # Rollout forward (bopl_rollout.cu): BOPL_BATCHED_FWD=1
    # (default) runs one grouped chain per buffer over the live rows with
    # per-bank weights (0 = the per-bank cuBLAS chain); BOPL_TF32=1 also picks
    # its tensor-core GEMM (0 = fp32 FMAs, bit-identical across runs);
    # BOPL_ZERO_COPY=0|1|2 how env buffers reach the kernels (0 = memcpys, the
    # fastest; 1/2 read mapped host memory), BOPL_ZERO_COPY_ACTIONS=1 writes
    # actions straight to host memory, BOPL_GRAPH_COPIES=1 (default) captures
    # the H2D/D2H memcpys into the buffer's CUDA graph, BOPL_OBS_SPARSE=1
    # (default) stages only the populated obs slots. BOPL_GPU_PROF=1 prints the
    # h2d/forward/d2h split of eval_gpu; BOPL_CUPTI_TRACE=<file> with
    # CUDA_INJECTION64_PATH=<cupti_profile.so> dumps a kernel timeline for
    # cupti_timeline.py.
    # OMP_NUM_THREADS below only governs the single-buffer synchronous step.
    for key, value in dict(BOPL_RL_GAME=game, BOPL_RL_BOOTSTRAP=ROOT/'native/fixtures/movement.rep',
                           BOPL_RL_LOG_DIR=output/'game-logs',
                           OMP_NUM_THREADS=os.environ.get('BOPL_OMP_THREADS', a.workers)).items():
        os.environ[key] = str(value)
    (output/'game-logs').mkdir(exist_ok=True)
    os.environ['BOPL_NATIVE_ENCODER'] = a.encoder
    # Policy v4 wanted the whole GPU shared on stream priority: its training
    # pass and its rollout chain are both several times v3's, and on the 24/24
    # green-context split of 'auto' each is SM-bound (v4-lite, w128 b4: train
    # 0.107 -> 0.090 s, rollout 0.120 -> 0.114 s, +5% sps when shared; 32/16
    # and 16/32 are worse either way). Since the GEMM, sets and encode work of
    # 2026-10-08 the spec shape (w256 b4) has the training pass short enough
    # for 16 SMs: 32/16 beats shared in 3 of 3 rounds (1.142M vs 1.058M,
    # 1.150M vs 1.089M, 0.948M vs 0.935M sps; train_s 0.052 vs 0.044 s
    # against 0.057 s of rollout wall), 40/8 is train-bound (0.093 s, 0.69M).
    # If the rollout gets much faster, training becomes the wall again:
    # re-measure 32 against 0. The serial trainer (BOPL_ASYNC=0) keeps the
    # whole GPU: an explicit count would still partition it. v3 keeps the
    # measured 'auto'. Scheduling only: the numbers do not change.
    serial = os.environ.get('BOPL_ASYNC', '1').strip() == '0'
    os.environ.setdefault('BOPL_GPU_SPLIT', ('0' if serial else '32') if a.encoder == 'v4' else 'auto')
    v4_shape = a.v4_shape
    if a.encoder == 'v4':
        for name, (variable, key) in V4_SHAPE.items():
            os.environ[variable] = str(v4_shape[key])
        if a.v4_row_budget:
            os.environ['BOPL4_ROW_BUDGET'] = a.v4_row_budget
        os.environ['BOPL4_FULL_ROWS'] = a.v4_full_rows
        layout.policy_v4_weights(a.hidden, a.layers, aim_fourier=a.aim_fourier, aim_ar=a.aim_ar, **v4_shape)  # rejects unaligned shapes
    os.environ['BOPL_ENT_HEAD_SCALE'] = ','.join(repr(s) for s in a.head_scale)
    os.environ['BOPL_AIM_FOURIER'] = str(a.aim_fourier)  # read once by the trainer (bopl_aim_config)
    os.environ['BOPL_AIM_AR'] = str(a.aim_ar)
    if a.dump_rows is not None:
        # The env workers write the terrain and rays the GPU writes too
        # (bopl_puffer.h bopl_pack_rows), so the transport check compares them.
        os.environ['BOPL_RAYS_CHECK'] = '1'
    import numpy as np
    import torch
    from pufferlib import _C, pufferl
    if not _C.gpu:
        raise RuntimeError('Native training requires the CUDA build')
    install_loadout_set(_C,a.loadouts['lineups'])
    install_loadout_weights(_C,a.loadouts.get('weights',[]))
    kit_lib = bank_kits.library(_C) if a.bank_kits else None
    if a.bank_kits:
        bank_kits.install_bot_kit(kit_lib, a.bot_kit or [0, 0, 0])
    style = install_style(_C, a.style_program)
    if a.style_program is not None:
        (output/'style.bin').write_bytes(a.style_program.data)
        (output/'style.bin.txt').write_text(a.style_program.dump())
    print('Effective loadouts: '+json.dumps(a.loadouts,separators=(',',':')),flush=True)
    # A binding compiled from another checkout (a rift's copied venv finder) or before
    # the current env/kernel sources trains silently on stale code; fail instead.
    binding = Path(_C.__file__).resolve()
    if not binding.is_relative_to(ROOT):
        raise RuntimeError(f'pufferlib._C resolves to {binding}, outside {ROOT}: fix the venv editable finder and rebuild')
    puffer_dir = ROOT/'train'
    sources = [puffer_dir/name for name in ('bopl_puffer.h', 'bopl_style.h', 'binding.c', 'bopl_rollout.cu', 'bopl_set_encoder.cu',
                                           'bopl_aim.h', 'bopl_layout.h', 'bopl_v4.cu', 'bopl_obs_v4.h', 'bopl_layout_v4.h',
                                           'pufferlib-bopl.patch')] + [ROOT/'sim/abi/bopl_rl.c']
    stale = [path.name for path in sources if path.exists() and path.stat().st_mtime > binding.stat().st_mtime]
    if stale:
        raise RuntimeError(f'pufferlib._C ({binding}) is older than {stale}; run train/build.py --device cuda')
    torch.cuda.init()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    # PufferLib parses sys.argv itself; the wrapper owns this command's arguments.
    sys.argv = [sys.argv[0]]
    args = pufferl.load_config('bopl')
    vector = NativeVector(_C, a.workers, a.buffers, a.banks, a.historical_every, a.team_every)
    learner_batch = vector.learner_rows * a.horizon
    if a.minibatch_div < 1 or vector.learner_rows % a.minibatch_div:
        raise SystemExit(f'--minibatch-div {a.minibatch_div} must divide the {vector.learner_rows} learner rows')
    minibatch = learner_batch // a.minibatch_div
    native_batch = vector.total_agents * a.horizon
    updates = math.ceil(a.steps / learner_batch)
    # Native replay_ratio multiplies padded rows. Correct for those rows, and
    # round up one float32 ULP because C truncates the resulting update count.
    ratio = float(np.nextafter(np.float32(a.learner_passes * learner_batch / native_batch), np.float32(np.inf)))
    args.update(seed=a.seed, reset_state=False, cudagraphs=a.cudagraphs)
    # Compact layout (NativeVector.compact): each bank's slice holds only its
    # live opponent rows and primary_rows fixes the learner slice, so the
    # trainer's bank_layout ends before the idle tail.
    args['vec'].update(total_agents=vector.total_agents, num_buffers=a.buffers,
                       num_threads=a.workers, num_frozen_banks=a.banks,
                       frozen_bank_pct=(vector.live_per_bank + .000001) / vector.agents_per_buffer,
                       primary_rows=vector.primary_rows)
    if a.kl_reference:
        args['vec'].update(kl_reference=1, kl_coef=a.kl_coef)
    args['policy'].update(hidden_size=a.hidden, num_layers=a.layers)
    args['train'].update(total_timesteps=updates*native_batch, seed=a.seed,
                         horizon=a.horizon, minibatch_size=minibatch, replay_ratio=ratio,
                         learning_rate=a.learning_rate, anneal_lr=a.anneal_lr, min_lr_ratio=a.min_lr_ratio,
                         ent_coef=a.entropy, anneal_ent_coef=0,
                         min_ent_coef_ratio=a.entropy_floor/a.entropy)
    if a.ema_decay:
        args['train'].update(ema_decay=a.ema_decay)
    if a.optimizer == 'adam':
        args['train'].update(adam=1)
    if a.optimizer == 'muon-adamw':
        names = [name for name, _ in layout.policy_v4_tensors(a.hidden, a.layers, aim_fourier=a.aim_fourier,
                                                              aim_ar=a.aim_ar, **a.v4_shape)]
        args['train'].update(adam=2, adam_regs=sum(1 << i for i, name in enumerate(names) if name in ADAMW_TENSORS),
                             adam_lr_scale=a.adam_lr/a.learning_rate)
    if a.weight_decay:
        args['train'].update(weight_decay=a.weight_decay)
    if a.gae_lambda is not None:
        args['train'].update(gae_lambda=a.gae_lambda)
    if a.gamma is not None:
        args['train'].update(gamma=a.gamma)
    if a.prio_alpha is not None:
        args['train'].update(prio_alpha=a.prio_alpha)
    # Scripted banks are the LAST ones. When share_min < 1 some of their matches
    # fall to frozen weights, so the pool manages every bank (bot matches stay
    # out of hist_*_bank). At share_min = share_max = 1 the leftover never
    # happens and the pool skips those banks, as in the old no-decay path --
    # unless every bank is scripted, where the pool keeps one (idle) bank
    # because the pool state is defined over at least one.
    # With kit pinning a scripted bank's matches on space maps go to its frozen
    # weights too, so the pool manages it whatever the share.
    policy_banks = a.banks if (a.n_scripted and (a.scripted_share_min < 1.0 or a.bank_kits)) else a.banks - a.n_scripted
    policy_banks = max(1, policy_banks)
    bank_groups = [g for g in a.bank_groups.split(',') if g] or ['all'] * policy_banks
    if len(bank_groups) != policy_banks or any(g not in native_selfplay.GROUPS for g in bank_groups):
        raise SystemExit(f'--bank-groups needs {policy_banks} of {native_selfplay.GROUPS}, got {a.bank_groups!r}')
    if 'pinned' in bank_groups and not a.pinned_opponents:
        raise SystemExit('a `pinned` entry in --bank-groups needs --pinned-opponents')
    scripted_first_bank = a.banks - a.n_scripted
    scripted_kinds = 0
    for i in range(a.n_scripted):
        if a.scripted_bots:
            kind_name = a.scripted_bots[i]
        else:
            kind_name, _, _, _ = scripted_roster.split_name(a.scripted_roster[0])
        scripted_kinds |= SCRIPTED_KINDS[kind_name] << 4 * (scripted_first_bank + i)
    map_chunks=map_mask_chunks(a.training_maps)
    pool_mask=sum(1 << ability for ability in a.loadouts['pool'])
    args['env'].update(players=2, banks=a.banks, historical_every=a.historical_every, seed=a.seed,
                       **(dict(team_every=a.team_every) if a.team_every else {}),
                       action_repeat=4, observation_delay=9, map=a.training_maps[0],
                       # Four 31-bit chunks: binding options travel as doubles.
                       training_map_mask=map_chunks[0], training_map_mask_hi=map_chunks[1],
                       training_map_mask_2=map_chunks[2], training_map_mask_3=map_chunks[3],
                       ability1=a.loadouts['fixed'][0], ability2=a.loadouts['fixed'][1], ability3=a.loadouts['fixed'][2],
                       loadout_pool=pool_mask&0x7fffffff, loadout_pool_hi=(pool_mask>>31)&0x7fffffff,
                       loadout_no_dup=int(a.loadouts['no_dup']),
                       **(dict(loadout_learner_first=1) if a.loadouts.get('learner_first') else {}),
                       **(dict(bank_kits=1) if a.bank_kits else {}),
                       mirror_share_ppm=round(a.loadouts['mirror_share']*1_000_000),
                       max_episode_ticks=a.episode_ticks, stocks=a.stocks, reward_kill_milli=round(a.reward_kill*1000), reward_timeout_milli=round(a.reward_timeout*1000),
                       reward_self_death_milli=round(a.reward_self_death*1000),
                       scripted_kinds=scripted_kinds, scripted_level=SCRIPTED_LEVELS[a.scripted_level],
                       obs_layout=layout.V4_LAYOUT_VERSION if a.encoder == 'v4' else 3)
    pufferl.validate_config(args)
    # Physical decisions per learner decision: both players of every game per
    # step over the learner rows, 4/3 at historical_every 2 and 2 at 1.
    target_physical = math.ceil(a.steps * (2 * a.historical_every) / (2 * a.historical_every - 1))
    if a.team_every:   # team games add learner rows, not frozen ones: one opponent row per historical game
        target_physical = math.ceil(a.steps * (vector.learner_rows + a.workers // 2) / vector.learner_rows)
    # The pool counts native agent steps (padded rows included);
    # the knobs above count learner decisions.
    native_per_learner = native_batch / learner_batch
    pool_config = dict(backend='league.pfsp', banks=policy_banks, historical_games_per_bank=vector.historical_per_bank,
                    max_size=a.pool_size, pfsp_p=a.pfsp_p, min_games=a.min_games,
                    snapshot_interval=a.snapshot_interval, opponent_timeout=a.opponent_timeout,
                    seed=a.pool_seed, step_unit='learner_decisions', bank_groups=bank_groups,
                    **({'pfsp_f': a.pfsp_f} if a.pfsp_f != 'hard' else {}),
                    native_steps_per_learner_decision=native_per_learner, mid_match_swaps='counted in neither bank statistic')
    obs_layout_version = layout.V4_LAYOUT_VERSION if a.encoder == 'v4' else 3
    obs_size = layout.obs_size_of(obs_layout_version)
    architecture = {'set': layout.ARCHITECTURE, 'v4': layout.V4_ARCHITECTURE}.get(a.encoder, 'puffernet-mingru')
    config = dict(schema=1, algorithm='recurrent PPO', backend='pufferlib-native', map_catalogue='identity-124',
                  step_unit='physical_player_decisions', steps=target_physical, learner_steps=a.steps,
                  seed=a.seed, workers=a.workers, banks=a.banks, historical_every=a.historical_every,
                  team_every=a.team_every,
                  encoder=a.encoder, hidden=a.hidden, layers=a.layers,
                  scripted_bots=a.scripted_bots, scripted_level=a.scripted_level, scripted_roster=a.scripted_roster,
                  scripted_share_min=a.scripted_share_min, scripted_share_max=a.scripted_share_max,
                  scripted_banks=a.n_scripted,
                  **(dict(bank_kits=dict(pinned_opponents=a.pinned_opponents, bot_kit=a.bot_kit,
                                         scripted_elo=a.scripted_elo)) if a.bank_kits else {}),
                  training_maps=a.training_maps, evaluation_maps=a.evaluation_maps,
                  heldout_maps=a.heldout_maps, episode_ticks=a.episode_ticks, stocks=a.stocks,
                  loadouts=a.loadouts,
                  observation_delay_ticks=9, telemetry_schema=1,
                  reward=dict(win=1, loss=-1, timeout=a.reward_timeout, kill=a.reward_kill, death=-a.reward_kill,
                              self_death=-a.reward_self_death),
                  entropy=a.entropy, entropy_floor=a.entropy_floor, entropy_head_scale=a.head_scale,
                  entropy_free_rows=a.entropy_free_rows,
                  aim_fourier=a.aim_fourier, aim_ar=a.aim_ar, v4_shape=v4_shape,
                  entropy_schedule='exponential', entropy_half_life=a.entropy_half_life,
                  layout=dict(version=obs_layout_version, obs_size=obs_size, action_sizes=layout.ACTION_SIZES,
                              hold_lengths=layout.HOLD_LENGTHS),
                  learning_rate_schedule='cosine' if a.anneal_lr else 'constant', learner_passes=a.learner_passes,
                  gae_lambda=args['train']['gae_lambda'],
                  **(dict(ema_decay=a.ema_decay) if a.ema_decay else {}),
                  fixes=schedule_fixes(a),
                  **(dict(optimizer=a.optimizer) if a.optimizer != 'muon' else {}),
                  **(dict(adam_lr=a.adam_lr) if a.optimizer == 'muon-adamw' else {}),
                  **(dict(weight_decay=a.weight_decay) if a.weight_decay else {}),
                  selfplay=pool_config,
                  init_weights=checkpoint_identity(a.init_weights) if a.init_weights else None,
                  **(dict(kl_reference=checkpoint_identity(a.kl_reference), kl_coef=a.kl_coef) if a.kl_reference else {}),
                  frozen_opponents=frozen, evaluation_baseline_identity=baseline_identity, evaluation_baseline=str(a.evaluation_baseline.resolve()) if a.evaluation_baseline else None,
                  style=(dict(paths=[str(path.resolve()) for path in a.style], sha256=a.style_sha256,
                              terms=a.style_program.term_names, fade=list(a.style_program.fade),
                              fade_steps=a.style_fade_steps or a.steps, program='style.bin') if a.style_program is not None else None),
                  game=str(game), runtime=runtime, native=args,
                  native_binding_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest())
    if resume:
        # The original config stays the record of the run; only the budget and
        # the resume history change.
        config = previous_config
        config.update(steps=target_physical, learner_steps=a.steps)
        config.setdefault('resumed', []).append(dict(at_learner_steps=resume['counts']['learner_steps'],
                                                     checkpoint=resume['checkpoint'], steps=a.steps))
        if schedule_changes:
            # config.json keeps describing the run as it now trains; the
            # resume record keeps what it trained with before.
            config['resumed'][-1]['changed'] = schedule_changes
            config.update({name: new for name, (_, new) in schedule_changes.items()})
            config['native']['train'].update(ent_coef=a.entropy, min_ent_coef_ratio=a.entropy_floor/a.entropy)
            print(f'--resume: entropy schedule changed {schedule_changes}', flush=True)
        fixes = config.get('fixes', DEFAULT_FIXES)
        if fixes != schedule_fixes(a):
            config['resumed'][-1]['changed'] = {**config['resumed'][-1].get('changed', {}), 'fixes': [fixes, schedule_fixes(a)]}
            config['fixes'] = schedule_fixes(a)
            print(f'--resume: fixes changed {fixes} -> {schedule_fixes(a)}', flush=True)
    atomic_json(output/'config.json', config)
    started = time.monotonic()
    counts = offset_counters(empty_counters(), resume['counts'] if resume else None)
    trainer = None
    pending = []  # background evaluation (process, log) pairs
    rating = []   # background in-run league rating, at most one (process, log)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def status(state, **extra):
        record = dict(status=state, pid=os.getpid(), step=counts['game_steps'],
                    learner_steps=counts['learner_steps'], matches=counts['matches'],
                    elapsed=time.monotonic()-started, target_steps=target_physical,
                    target_learner_steps=a.steps, **extra)
        atomic_json(output/'status.json', record)
        if state in ('complete', 'stopped', 'failed'):
            atomic_json(output/'result.json', record)
    status('starting')
    saved = {}
    # An all-uniform roster needs no league games: the uniform bot is the random anchor, at 0;
    # nor does one whose every entry has a --scripted-elo (rusher-hard has no league rating).
    uniform_only = all(scripted_roster.split_name(name)[0] == 'uniform' or name in a.scripted_elo
                       for name in a.scripted_roster)
    roster_entries = (scripted_roster.entries_from_names(
        a.scripted_roster, {**({} if uniform_only else scripted_roster.league_ratings()), **a.scripted_elo})
        if a.scripted_roster else [])
    # Resolved before the trainer exists so a typo'd member name, a missing
    # checkpoint or a member from another policy architecture fails here rather
    # than a minute into a run - or, for the architecture, never.
    run_policy = dict(architecture=architecture, obs_size=obs_size, hidden=a.hidden,
                      num_layers=a.layers, aim_fourier=a.aim_fourier, aim_ar=a.aim_ar, v4_shape=v4_shape)
    if a.opponents == 'self':
        league_opponents = []
    else:
        names = None if a.opponents == 'league' else [s for s in a.opponents.split(',') if s]
        league_opponents = registry.opponents(registry.load(), names, policy=run_policy)
    # Pinned members (bank_kits.py): on their own kits wherever drawn, and with
    # a `pinned` bank drawn only there (native_selfplay.candidates).
    pinned_rows, pinned_kits = {}, {}
    if a.pinned_opponents:
        league_table = registry.load()
        by_name = registry.by_name(league_table)
        if a.pinned_opponents == ['auto']:
            pinned = [o for o in league_opponents if registry.kits(by_name[o['name']])]
        else:
            pinned = registry.opponents(league_table, a.pinned_opponents, policy=run_policy)
        if 'pinned' in bank_groups and not pinned:
            raise SystemExit('a `pinned` bank but no --opponents member declares kits to pin')
        for member in pinned:
            kits = registry.kits(by_name[member['name']])
            if not kits:
                raise SystemExit(f"--pinned-opponents: {member['name']} declares no kits of its own to pin")
            pinned_kits[member['path']] = kits
            pinned_rows[member['path']] = bank_kits.member_rows(kits)
        pinned_paths = {member['path'] for member in pinned}
        league_opponents = [o for o in league_opponents if o['path'] not in pinned_paths] + \
                           [dict(member, pinned=True) for member in pinned]
    for flag, checkpoint in (('--init-weights', None if resume else a.init_weights), ('--kl-reference', a.kl_reference)):
        if not checkpoint:
            continue
        # The same flat-array hazard registry.opponents guards a bank against: a
        # mismatched checkpoint loads without error and trains from garbage.
        run_shape = dict(architecture=architecture, obs_size=obs_size, hidden=a.hidden,
                         num_layers=a.layers, aim_fourier=a.aim_fourier, aim_ar=a.aim_ar,
                         v4_shape=v4_shape)
        # Writer revisions (registry.WRITER_REVISIONS) load across: that is how a
        # v4.1 net is fine-tuned on v4.2 rows.
        differs = registry.shape_differs(registry.policy_shape(checkpoint), run_shape)
        if differs:
            raise SystemExit(f'{flag} {checkpoint}: architecture differs from the run: {differs}')
    print(f'league opponents: {len(league_opponents)}'
          + (' (' + ', '.join(o['name'] + ('[pinned]' if o.get('pinned') else '') for o in league_opponents) + ')'
             if league_opponents else ''),
          flush=True)
    if a.bank_kits:
        print('kit pinning: ' + json.dumps(dict(bank_groups=bank_groups, bot_kit=a.bot_kit,
                                                 pinned={o['name']: pinned_kits[o['path']] for o in league_opponents
                                                         if o.get('pinned')}), separators=(',', ':')), flush=True)
    share = 0.0
    try:
        trainer = _C.create_pufferl(args)
        vector.attach(trainer)
        # Bank swaps go through the backend: with kit pinning it installs the
        # bank's kit rows for the member it loads (bank_kits.PinnedBackend).
        backend = bank_kits.PinnedBackend(_C, kit_lib, pinned_rows) if a.bank_kits else _C
        if v4_shape is not None:
            # Each frozen member writes its own rows (a v4.1 member in a v4.2 run).
            backend = bank_kits.WriterBackend(backend, kit_lib or bank_kits.library(_C), v4_shape['ablate'])
        kit_audit = (bank_kits.Audit(kit_lib, vector.gpu_obs_ptr, a.workers, backend, pinned_kits,
                                     {o['path']: o['name'] for o in league_opponents}, a.bot_kit,
                                     registry._space_maps())
                     if a.bank_kits else None)
        if a.kl_reference:
            _C.load_reference_bank(trainer, str(a.kl_reference.resolve()))
        pool = ledger = None
        update_index = 0
        elapsed_before = 0.
        def persist_roster():
            if roster_entries:
                atomic_json(output/'pool/roster.json', scripted_roster.state_dict(
                    roster_entries, share=share, learner_elo=scripted_roster.LEARNER_ELO))
        def sync_scripted():
            nonlocal share
            if not roster_entries:
                return
            origin = scripted_roster.LEARNER_ELO
            elos = [entry['elo'] for entry in roster_entries]
            share = scripted_roster.scripted_share(elos, origin, a.scripted_share_min, a.scripted_share_max)
            set_scripted_curriculum(_C, share, roster_entries, scripted_roster.draw_weights(elos, origin))
        def save():
            step = counts['game_steps']
            if step not in saved:
                path = output/'checkpoints'/f'policy-{step:09d}.bin'
                write_checkpoint(path, lambda name: _C.save_weights(trainer, name), obs_size=obs_size,
                                 hidden=a.hidden, num_layers=a.layers, architecture=architecture,
                                 game_steps=step, learner_steps=counts['learner_steps'], config=config,
                                 aim_fourier=a.aim_fourier, aim_ar=a.aim_ar, v4_shape=v4_shape)
                saved[step] = path
                if a.ema_decay:
                    write_checkpoint(path.with_name(f'{path.stem}-ema.bin'), lambda name: _C.save_ema(trainer, name),
                                     obs_size=obs_size, hidden=a.hidden, num_layers=a.layers, architecture=architecture,
                                     game_steps=step, learner_steps=counts['learner_steps'],
                                     config=dict(config, weights='ema'),
                                     aim_fourier=a.aim_fourier, aim_ar=a.aim_ar, v4_shape=v4_shape)
            if pool is not None:
                # Everything --resume needs besides the weights, written with
                # every checkpoint so the last one is always consistent. Only
                # the newest Muon momentum is kept: resume is its one reader.
                momentum = output/'pool'/f'momentum-{step:09d}.bin'
                _C.save_momentum(trainer, str(momentum))
                ema = output/'pool'/f'ema-{step:09d}.bin' if a.ema_decay else None
                if ema:
                    _C.save_ema(trainer, str(ema), state=True)
                atomic_json(output/'pool/state.json', pool_state_json(
                    pool, ledger, counts=counts, elapsed=elapsed_before + time.monotonic() - training_started,
                    global_step=trainer.global_step, update_index=update_index, checkpoint=saved[step],
                    momentum=momentum, ema=ema))
                for old in [*(output/'pool').glob('momentum-*.bin'), *(output/'pool').glob('ema-*.bin')]:
                    if old not in (momentum, ema):
                        old.unlink()
                persist_roster()
            return saved[step]
        training_started = time.monotonic()
        if resume:
            saved = {int(path.stem.split('-')[1]): path for path in sorted((output/'checkpoints').glob('policy-*.bin'))
                     if not path.stem.endswith('-ema')}
            initial = saved[min(saved)]
            _C.load_weights(trainer, resume['checkpoint'])  # also restarts the EMA at these weights
            # Runs saved before momentum was persisted resume with it cold.
            momentum = resume.get('momentum')
            if momentum and Path(momentum).is_file():
                _C.load_momentum(trainer, momentum)
            else:
                print('--resume: no saved Muon momentum; the optimiser restarts cold', flush=True)
            config['resumed'][-1]['momentum'] = bool(momentum and Path(momentum).is_file())
            if a.ema_decay:
                ema = resume.get('ema')
                kept = bool(ema and Path(ema).is_file() and previous_config.get('ema_decay') == a.ema_decay)
                if kept:
                    _C.load_ema_state(trainer, ema)
                else:
                    print(f'--resume: no saved EMA at decay {a.ema_decay}; it restarts at the resumed weights', flush=True)
                config['resumed'][-1]['ema'] = kept
            if previous_config.get('ema_decay', 0.) != a.ema_decay:
                config['resumed'][-1]['changed'] = {**config['resumed'][-1].get('changed', {}),
                                                    'ema_decay': [previous_config.get('ema_decay', 0.), a.ema_decay]}
                config['ema_decay'] = a.ema_decay
                config['native']['train'].pop('ema_decay', None)
                config['native']['train'].update(**({'ema_decay': a.ema_decay} if a.ema_decay else {}))
            trainer.global_step = resume['global_step']
            pool, ledger = load_pool_state(resume)
            if len(pool['banks']) != policy_banks:
                raise SystemExit(f"--resume: the pool manages {len(pool['banks'])} banks but "
                                 f'--banks/--scripted-bots now leave {policy_banks} policy banks')
            for bank, state in enumerate(pool['banks']):
                backend.load_frozen_bank(trainer, bank, state['cur_opp_path'])
            native_selfplay.groups(pool, bank_groups)
            joined, left = native_selfplay.reconcile(pool, ledger, league_opponents)
            config['resumed'][-1].update(opponents=[o['name'] for o in league_opponents],
                                         joined=joined, left=left, bank_groups=bank_groups)
            print(f'--resume: league members joined {joined or "none"}, left {left or "none"}', flush=True)
            # Scripted banks never swap: their GPU rows run the initial policy
            # (actions unread; the C environment drives those players).
            for bank in range(policy_banks, a.banks):
                backend.load_frozen_bank(trainer, bank, str(initial))
            update_index = resume['update_index']
            elapsed_before = resume['elapsed']
            if roster_entries:
                roster_path = output/'pool/roster.json'
                if not roster_path.is_file():
                    raise SystemExit('--resume: pool/roster.json is missing')
                scripted_roster.restore_counts(roster_entries, json.loads(roster_path.read_text()))
        else:
            if a.init_weights:
                _C.load_weights(trainer, str(a.init_weights.resolve()))
            initial = save()
            # Every bank starts from the initial policy; PFSP retires and
            # redraws them independently as each matchup settles.
            for bank in range(a.banks):
                backend.load_frozen_bank(trainer, bank, str(initial))
            (output/'pool').mkdir()
            pool = native_selfplay.state(output/'pool', str(initial), policy_banks, vector.historical_per_bank, trainer.global_step,
                                         max_size=a.pool_size, min_games=a.min_games,
                                         snapshot_interval=round(a.snapshot_interval*native_per_learner),
                                         opp_timeout_steps=round(a.opponent_timeout*native_per_learner),
                                         pfsp_p=a.pfsp_p, seed=a.pool_seed, league=league_opponents,
                                         bank_groups=bank_groups)
            ledger = native_selfplay.members(str(initial), league_opponents)
        pool['pfsp_f'] = a.pfsp_f
        if roster_entries:
            sync_scripted()
        style.scale(counts['learner_steps'], a.style_fade_steps or a.steps)
        vector.reset()
        _C.log(trainer)  # Warmup losses are not measured training updates.
        if vector.snapshot()['learner_steps'] or vector.snapshot()['game_steps']:
            raise RuntimeError('Setup unexpectedly counted game decisions')
        base_counts = resume['counts'] if resume else None
        counts = offset_counters(vector.snapshot(), base_counts)
        if not resume:
            save()  # pool/state.json for the initial checkpoint
        config['evaluation_baseline'] = str(a.evaluation_baseline.resolve() if a.evaluation_baseline else initial)
        config['evaluation_baseline_identity'] = checkpoint_identity(Path(config['evaluation_baseline']))
        atomic_json(output/'config.json', config)
        evaluated_steps = set()
        # Evaluation runs in the background: the trainer keeps rolling out while
        # evaluate.py plays the saved checkpoint (a blocking eval cost 22-33% of
        # trial wall time in the overnight-1 sweep). At most one is in flight;
        # the next eval, the final save and shutdown all wait for it.
        def finish_evaluation():
            while pending:
                process, log = pending.pop()
                code = process.wait()
                log.close()
                if code:
                    raise subprocess.CalledProcessError(code, process.args)
        def evaluate(checkpoint):
            if (checkpoint_identity(Path(config['evaluation_baseline'])) != config['evaluation_baseline_identity']
                    or opponents(frozen_manifest) != frozen):
                raise RuntimeError('Frozen evaluation checkpoint identity changed during training')
            finish_evaluation()
            command = [sys.executable, str(ROOT/'eval/evaluate.py'), '--run', str(output),
                       '--checkpoint', str(checkpoint), '--initial', config['evaluation_baseline'],
                       '--evaluation-maps', ','.join(map(str, a.evaluation_maps)),
                       '--heldout-maps', ','.join(map(str, a.heldout_maps)),
                       '--games-per-opponent', str(a.evaluation_games), '--heldout-games', str(a.heldout_games),
                       '--frozen-games', str(a.frozen_games), '--workers', str(a.evaluation_workers), '--game', str(game),
                       '--scripted-opponents', a.scripted_opponents]
            if frozen_manifest:
                command += ['--opponents', str(frozen_manifest)]
            log = (output/'evaluation.log').open('a')
            pending.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
            evaluated_steps.add(counts['game_steps'])
            rate_in_run(checkpoint)
        # In-run rating curve: every evaluation checkpoint is also rated against the
        # league (league candidate, source 'in-run', a small budget on 2 workers at
        # nice 19, CPU only) into <run>/league/<stem>.json, which the dashboard and
        # the scorer read; the scorer's full rating of the last checkpoints replaces
        # theirs. Non-blocking; skipped while the previous one still runs.
        def rate_in_run(checkpoint):
            if a.no_in_run_rating:
                return
            if rating and rating[0][0].poll() is None:
                print(f'in-run league rating of {checkpoint.name} skipped: the previous one is still running', flush=True)
                return
            for process, log in rating:
                log.close()
            rating.clear()
            command = ['nice', '-n', '19', sys.executable, str(ROOT/'league'), 'candidate', str(checkpoint),
                       '--games', str(a.in_run_games), '--first-pass-games', '2', '--workers', '2',
                       '--source', 'in-run']
            log = (output/'league.log').open('a')
            rating.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
        def finish_rating():
            for process, log in rating:
                if process.wait():
                    print(f'in-run league rating exited {process.returncode} (see league.log)', flush=True)
                log.close()
            rating.clear()
        next_checkpoint = (counts['learner_steps']//a.checkpoint_every+1)*a.checkpoint_every
        next_eval = (counts['learner_steps']//a.eval_every+1)*a.eval_every
        next_metrics = 0
        previous, previous_elapsed = counts, elapsed_before
        free, applied_scale = None, list(a.head_scale)
        # Overlap (BOPL_ASYNC=1 default; 0 restores the
        # serial trainer bit for bit): the native trainer keeps two rollout
        # slots, so each update collects the next batch on the buffer threads
        # while the GPU trains the previous one, and the update's wall time is
        # max(rollout, train) instead of their sum. The actor plays a snapshot
        # of the weights taken when its rollout starts (one update of policy
        # lag; the stored logprobs/values are that actor's, so the PPO ratios
        # are exact). Everything that reads what a rollout writes (env logs and
        # counters in _C.log/vector.snapshot, bank swaps in selfplay.step,
        # load_weights) runs after rollouts_finish. `counts` are the decisions
        # trained so far; `ready` those collected into the slot that trains
        # next, so a rollout that is collected but never trained (SIGTERM) is
        # not counted, and every update still accounts for exactly one batch.
        overlap = _C.num_rollout_slots(trainer) > 1
        more, ready, rollout_seconds = False, None, None
        def dump_rows():
            # The trainer's own decoder rows on its first observations, from
            # the weights the initial checkpoint holds (nothing has trained yet).
            nonlocal stop
            if a.dump_rows is None or stop:
                return
            if resume or 0 not in saved:
                raise RuntimeError('--dump-rows needs a fresh run with the initial checkpoint saved')
            import numpy as np
            observations, rows = _C.bopl_policy_rows(trainer, 0, 64)
            # Packed observation transport (bopl_obs_pack.h): floats of any unpacked
            # device row that differed from its host row over the rollout (-1: dense).
            transport_mismatches = _C.bopl_transport_mismatches(trainer)
            # The batched rollout chain (bopl_rollout.cu) on the same rows: the
            # kernels the rollout actually runs, which bopl_policy_rows does not.
            extra = {}
            if architecture == layout.V4_ARCHITECTURE:
                extra['chain_rows'] = _C.bopl_chain_rows(trainer, observations)
                # Every step's learner rows of the rollout, for the bit-for-bit
                # compare of two builds' chains (a chain call takes one step's rows).
                per_call = observations.shape[0]
                def chain_all(obs_all):
                    return np.concatenate([_C.bopl_chain_rows(trainer, obs_all[i:i + per_call])
                                           for i in range(0, obs_all.shape[0], per_call)])
                extra['chain_observations'] = np.concatenate([_C.bopl_policy_rows(trainer, t, 64)[0] for t in range(a.horizon)])
                extra['chain_rows_all'] = chain_all(extra['chain_observations'])
                if a.chain_reference is not None:
                    reference = np.load(a.chain_reference)
                    _C.load_weights(trainer, str(reference['checkpoint']))
                    extra['chain_rows_reference'] = chain_all(reference['chain_observations'])
            a.dump_rows.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.dump_rows, observations=observations, rows=rows, checkpoint=str(saved[0].resolve()),
                     obs_size=obs_size, architecture=architecture, transport_mismatches=transport_mismatches, **extra)
            print(f'wrote {a.dump_rows}: {rows.shape[0]} rows of {rows.shape[1]} from {saved[0].name}; '
                  f'transport mismatches {transport_mismatches}', flush=True)
            stop = True
        if a.dump_rows is not None:
            _C.bopl_transport_check(trainer, True)
        if overlap:
            rollout_seconds = _C.rollouts(trainer)  # bootstrap: the first slot before the loop
            ready = offset_counters(vector.snapshot(), base_counts)
            dump_rows()
        training_started = time.monotonic()
        deferred = None   # the last update's bookkeeping, not done yet (the loop's end)
        # BOPL_TRAIN_AB_DEFER=<period> (measurement only): updates alternate
        # every <period> between the deferred bookkeeping (arm 0) and the
        # serial order (arm 1, before the next rollout starts); the outputs
        # are the same either way, the arms' wall_seconds are the A/B.
        defer_ab = int(os.environ.get('BOPL_TRAIN_AB_DEFER', '0'))
        def bookkeep(p, evaluate_due=True):
            nonlocal next_checkpoint, next_eval, next_metrics
            done = p['counts']   # == counts: the next rollout is not collected yet
            logged = done['learner_steps'] >= min(next_metrics, a.steps) or stop
            if logged and a.metrics_every:
                next_metrics = (done['learner_steps']//a.metrics_every+1)*a.metrics_every
            native_selfplay.register(pool, ledger, done['game_steps'], done['learner_steps'])
            after_pool = native_selfplay.summary(pool, ledger)
            native = p['native']
            record = update(native, done, p['previous'], elapsed=p['elapsed'], previous_elapsed=p['previous_elapsed'],
                            entropy_coefficient=native['loss']['entropy_coefficient'],
                            learning_rate=native['loss']['learning_rate'],
                            pool_opponents=sorted(bank['opponent'] for bank in after_pool['banks']),
                            rollout_seconds=p['rollout_seconds'], train_seconds=p['train_seconds'])
            # With overlap the rollout (timed on the buffer threads) and the
            # train phase run concurrently, so they no longer sum to the
            # update's wall time; wall_seconds is that time (previous
            # record to this one), which is what sps/learner_sps divide by.
            record['wall_seconds'] = p['elapsed']-p['previous_elapsed']
            if 'prio_beta' in native['loss']:
                record['prio_beta'] = native['loss']['prio_beta']
            record['overlap'] = overlap
            record['pool'] = after_pool
            if p['scripted'] is not None:
                record['scripted'] = p['scripted']
                record['scripted_share'] = p['share']
            record['free_fraction'] = p['free']
            if p['style'] is not None:
                # Per term in full, plus flat keys the board charts: the fade scale,
                # the reward paid per learner decision, and each term's share of it.
                record['style'] = p['style']
                record['style_scale'] = p['style']['scale']
                record['style_paid'] = p['style']['paid_per_decision']
                for name, term in p['style']['terms'].items():
                    record['style/' + name] = p['style']['scale'] * term['pay_per_decision']
            record['entropy_head_scale_applied'] = p['applied_scale']
            if p['bank_kits'] is not None:
                # What each bank's frozen seat holds, read between rollouts (bank_kits.Audit).
                record['bank_kits'] = p['bank_kits']
            # The native trainer's own per-update timers (seconds, averaged
            # over buffers): eval_gpu = inference graph + action copy-back,
            # eval_env = OpenMP game stepping, rollout = whole barrier.
            record['native_perf'] = {key: native['perf'].get(key) for key in ('rollout', 'eval_gpu', 'eval_env')}
            packing = p['packing']
            if packing is not None:
                # The last training forward's packing (players, slots, entities, slimes): rows the packed set
                # buffers dropped (kept past the budget under --v4-full-rows), and the rows packed per batch row
                # (the row budget the batch needed).
                record['set_overflow'] = packing[:4]
                if len(packing) >= 9 and packing[8]:
                    record['set_rows'] = [round(n / packing[8], 3) for n in packing[4:8]]
                if len(packing) == 13 and any(packing[9:13]):
                    record['set_cut'] = packing[9:13]   # --v4-full-rows caps: rows cut from single samples
            if logged:
                metrics.write(json.dumps(record, allow_nan=False)+'\n'); metrics.flush()
            for event in native_selfplay.events(p['before_pool'], after_pool, ledger, done['learner_steps']):
                events.write(json.dumps(event, allow_nan=False)+'\n')
            events.flush()
            saturation = after_pool['pfsp']
            # Null until the pool has drawn/played anything; see _saturation.
            digits = lambda value: None if value is None else round(value, 3)
            console = dict(step=record['step'], learner_steps=record['learner_steps'],
                           learner_sps=record['learner_sps'], matches=record['matches'],
                           pool=after_pool['size'],
                           # The saturation line: how concentrated the curriculum
                           # actually is, next to what the pool currently intends.
                           pfsp=dict(f=saturation['f'], serving=saturation['serving'],
                                     top2=digits(saturation['played_top2']),
                                     played_entropy=digits(saturation['entropy']['played']),
                                     intended_entropy=digits(saturation['entropy']['intended']),
                                     divergence=digits(saturation['divergence'])))
            if p['bank_kits'] is not None:
                console['bank_kits'] = {bank: dict(member=row['member'], pinned=row['member_matches'],
                                                   bot=row['bot'], earlier=row['earlier'], off_kit=row['off_kit'])
                                        for bank, row in p['bank_kits']['banks'].items()
                                        if row['pinned'] or row['bot'] or row['off_kit']}
            if p['scripted'] is not None:
                console['scripted'] = dict(
                    share=round(p['share'], 3),
                    **{name: (None if row['score_rate'] is None else round(row['score_rate'], 3))
                       for name, row in record['scripted'].items()})
            if logged:
                print(json.dumps(console), flush=True)
            if done['learner_steps'] >= next_checkpoint:
                save()
                next_checkpoint = (done['learner_steps']//a.checkpoint_every+1)*a.checkpoint_every
            status('running')
            if evaluate_due and not a.no_eval and done['learner_steps'] >= next_eval and not stop:
                evaluate(save())
                next_eval = (done['learner_steps']//a.eval_every+1)*a.eval_every
        def flush_deferred(failing=False):
            nonlocal deferred, counts
            if deferred is None:
                return
            p, deferred = deferred, None
            if not failing:
                bookkeep(p)
                return
            # A failure after an update's critical half: write what the serial
            # order would have written already (save and status read counts), best effort.
            current, counts = counts, p['counts']
            try:
                bookkeep(p, evaluate_due=False)
            except Exception:
                pass
            finally:
                counts = current
        with (output/'metrics.jsonl').open('a') as metrics, (output/'pool.jsonl').open('a') as events, \
                contextlib.ExitStack() as on_exit:
            on_exit.callback(flush_deferred, True)   # a no-op unless the loop failed
            while counts['learner_steps'] < a.steps and not stop:
                before = time.monotonic()
                if overlap:
                    # Another update follows this one: collect its batch now.
                    more = ready['learner_steps'] < a.steps
                    if roster_entries:
                        # Matches starting during the next rollout draw against this roster.
                        sync_scripted()
                    if more:
                        style.scale(ready['learner_steps'], a.style_fade_steps or a.steps)
                        _C.rollouts_start(trainer)
                else:
                    style.scale(counts['learner_steps'], a.style_fade_steps or a.steps)
                    rollout_seconds = _C.rollouts(trainer)
                    dump_rows()
                flush_deferred()   # the previous update's bookkeeping, while this rollout runs
                rolled = time.monotonic()
                # Configure the native loss before training. Native train_impl
                # copies this scalar into a device buffer each update, so CUDA
                # graph replay observes the new coefficient. Count only the
                # learner decisions completed before this rollout; warmup and
                # frozen/padded rows never advance the schedule.
                trainer.hypers.ent_coef = entropy_coefficient(
                    counts['learner_steps'], a.entropy, a.entropy_floor, a.entropy_half_life)
                trainer.hypers.prio_beta = prio_beta(a, trainer.hypers.prio_beta0, counts['learner_steps'])
                if a.anneal_lr_fix:
                    trainer.hypers.lr_progress = min(1.0, counts['learner_steps'] / a.steps)
                # Latched rows contribute zero entropy and zero entropy gradient
                # to the button and hold heads, and the loss is a mean over all
                # rows, so with most decisions latched the bonus on those heads
                # is diluted by the free fraction (runs 005/006 collapsed into a
                # 16-decision latch this way). Dividing the head scale by the
                # previous rollout's free fraction makes it a mean over free rows.
                if a.entropy_free_rows and free is not None:
                    applied_scale = [s / free if h in LATCHED_HEADS else s for h, s in enumerate(a.head_scale)]
                    _C.set_ent_head_scale(trainer, [min(100.0, s) for s in applied_scale])
                _C.train(trainer)
                trained = time.monotonic()
                if overlap and more:
                    rollout_seconds = _C.rollouts_finish(trainer)
                native = _C.log(trainer)
                style_record = style.read(vector)   # no rollout in flight here
                kit_record = kit_audit() if kit_audit else None   # nor here: what each frozen seat holds
                free = free_fraction(native['env']) if a.entropy_free_rows else None
                if overlap:
                    counts = ready
                    if more:
                        ready = offset_counters(vector.snapshot(), base_counts)
                else:
                    counts = offset_counters(vector.snapshot(), base_counts)
                if (counts['learner_steps']-previous['learner_steps'] != learner_batch
                        or counts['game_steps']-previous['game_steps'] != vector.physical_per_step*a.horizon):
                    raise RuntimeError('Native rollout physical/learner decision accounting mismatch')
                elapsed = elapsed_before + time.monotonic()-training_started
                # Snapshots, PFSP draws and bank retirements from this rollout's finished games.
                before_pool = native_selfplay.summary(pool, ledger, saturation=False)  # feeds events() only
                flat = {'env/'+key: value for key, value in native['env'].items()}
                native_selfplay.credit(pool, flat, ledger)
                native_selfplay.step(trainer, backend, pool, ledger, update_index)
                update_index += 1
                scripted = None
                if roster_entries:
                    n = float(flat.get('env/n', 0.0))
                    window = []
                    for i in range(len(roster_entries)):
                        games = round(float(flat.get(f'env/bot_n_entry_{i}', 0.0)) * n)
                        wins = round(float(flat.get(f'env/bot_score_entry_{i}', 0.0)) * n)
                        decided = round(float(flat.get(f'env/bot_decided_entry_{i}', 0.0)) * n)
                        # A timeout and a mutual death are draws against the
                        # entry, so they belong in games with half credit, not
                        # dropped from the denominator.
                        window.append((games, wins, max(0, games - decided)))
                    scripted_roster.apply_window(roster_entries, window,
                                                 scripted_roster.LEARNER_ELO, scripted_roster.ELO_K)
                    sync_scripted()
                    weights = scripted_roster.draw_weights([e['elo'] for e in roster_entries],
                                                           scripted_roster.LEARNER_ELO)
                    scripted = {}
                    for i, entry in enumerate(roster_entries):
                        games = entry['games']
                        scripted[entry['name']] = dict(
                            bank=scripted_first_bank, elo=entry['elo'], games=games,
                            wins=entry['wins'], draws=entry['draws'],
                            winrate=(entry['wins']/games if games else None),
                            score_rate=scripted_roster.score_rate(entry), weight=weights[i],
                            window_games=window[i][0], window_wins=window[i][1],
                            window_draws=window[i][2])
                # The rest of the update (register, the pool summary, the
                # metrics and pool.jsonl lines, the console line, checkpoint,
                # status, evaluation) reads nothing the next rollout writes
                # and writes nothing it reads, so it waits for that rollout to
                # start (bookkeep, at the top of the next iteration or after
                # the loop) instead of holding it back ~0.8 ms. What it reads
                # that a later call changes is taken here.
                deferred = dict(native=native, counts=counts, previous=previous, elapsed=elapsed,
                                previous_elapsed=previous_elapsed, rollout_seconds=rollout_seconds,
                                train_seconds=trained-rolled, before_pool=before_pool, scripted=scripted,
                                share=share, free=free, applied_scale=applied_scale,
                                packing=list(_C.bopl_v4_overflow()) if a.encoder == 'v4' else None,
                                style=style_record, bank_kits=kit_record)
                previous, previous_elapsed = counts, elapsed
                if defer_ab and (update_index // defer_ab) & 1:
                    flush_deferred()   # BOPL_TRAIN_AB_DEFER's arm 1: the serial order
            flush_deferred()
            final = save()
            if not a.no_eval and not stop and counts['game_steps'] not in evaluated_steps:
                evaluate(final)
            finish_evaluation()
            finish_rating()
        status('stopped' if stop else 'complete', checkpoint=str(final))
    except BaseException as error:
        for process, log in pending + rating:
            process.terminate()
            log.close()
        status('failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        if trainer is not None:
            _C.close(trainer)


if __name__ == '__main__':
    main()
