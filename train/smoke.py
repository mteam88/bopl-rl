"""Native PufferLib training smoke: routing, frozen banks, real updates, counters, checkpoints.

Builds a mixed vector (self-play games and historical games whose opponent rows
come from frozen weight banks), applies the packed agent permutation the native
trainer's banks slice, trains a few real updates on the GPU and checks after
each one that the host observation rows still describe paired players, that
idle bank rows stay blank, that the cumulative counters match the exact number
of decisions the games took, that only learner rows were learned from, and that
a checkpoint round-trips through both the GPU trainer and the CPU evaluator.
Finally a frozen bank is replaced by the trained weights (pool rotation) and one
more update runs against it. With --games 16 --updates 20 --episode-ticks 3600
it doubles as a first native throughput read (learner decisions per second are
printed), but the qualified numbers come from the trainer benchmark.
"""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/'native/tools'), str(ROOT/'eval'), str(HERE)]
import reference  # noqa: E402
import counters  # noqa: E402
from check_routing import layout, build_perm, OBS, ROW, PLAYER0, PLAYER  # noqa: E402
import layout as policy_layout  # noqa: E402  (eval/layout.py: architecture string, heads)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--output', type=Path, default=ROOT/'native/artifacts/rl/native-smoke')
parser.add_argument('--games', type=int, default=4, help='games; every second one is historical (multiple of 2*buffers)')
parser.add_argument('--buffers', type=int, default=2)
parser.add_argument('--banks', type=int, default=2, help='frozen opponent banks per historical game')
parser.add_argument('--updates', type=int, default=3)
parser.add_argument('--graphs', type=int, default=1, help='CUDA graph warmup epochs (>=1) or -1 for eager')
parser.add_argument('--horizon', type=int, default=64)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--episode-ticks', type=int, default=120)
parser.add_argument('--warmup-ticks', type=int, default=60)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--seed', type=int, default=73)
parser.add_argument('--game', type=Path)
parser.add_argument('--encoder', choices=('linear', 'set'), default='linear', help='native encoder (sets BOPL_NATIVE_ENCODER before the trainer is created)')
parser.add_argument('--aim-fourier', type=int, default=0, help='Fourier aim head harmonics (BOPL_AIM_FOURIER); 0 = raw bins')
parser.add_argument('--aim-ar', type=int, choices=(0, 1), default=0, help='aim head conditioned on the sampled buttons (BOPL_AIM_AR)')
options = parser.parse_args()
os.environ['BOPL_NATIVE_ENCODER'] = options.encoder
os.environ['BOPL_AIM_FOURIER'] = str(options.aim_fourier)
os.environ['BOPL_AIM_AR'] = str(options.aim_ar)
ARCHITECTURE = policy_layout.ARCHITECTURE if options.encoder == 'set' else 'puffernet-mingru'
if options.games % (2*options.buffers) or options.banks < 1 or options.updates < 1 or options.graphs == 0:
    parser.error('games must be a multiple of 2*buffers, banks and updates positive, graphs >= 1 or -1')
output = options.output.resolve()
output.mkdir(parents=True, exist_ok=False)
os.environ['BOPL_RL_GAME'] = str(reference.resolve_game(options.game))
os.environ['BOPL_RL_BOOTSTRAP'] = str(ROOT/'native/fixtures/movement.rep')
os.environ['BOPL_RL_LOG_DIR'] = str(output/'logs'); (output/'logs').mkdir()
os.environ.setdefault('OMP_NUM_THREADS', str(options.threads))
sys.argv = [sys.argv[0]]  # pufferl.load_config parses the command line

import torch  # noqa: E402  (CPU evaluator reload only)
from pufferlib import _C, pufferl  # noqa: E402
from native_checkpoint import write_checkpoint  # noqa: E402
from native_metrics import update as metrics_update, empty_counters  # noqa: E402
from native_policy import load as load_policy  # noqa: E402

if not _C.gpu:
    raise RuntimeError('Build the CUDA backend first (build.py --device cuda)')
torch.set_num_threads(1)
lib = counters.bind(C.CDLL(_C.__file__))
lib.bopl_vec_handles.argtypes = [C.POINTER(C.c_void_p), C.c_int]; lib.bopl_vec_handles.restype = C.c_int
lib.bopl_vec_reset.argtypes = [C.c_void_p]; lib.bopl_vec_reset.restype = C.c_int
lib.bopl_vec_host.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
lib.bopl_vec_host.restype = C.c_int
lib.bopl_vec_layout.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int]; lib.bopl_vec_layout.restype = C.c_int

games, buffers, banks, horizon = options.games, options.buffers, options.banks, options.horizon
historical = games//2
rows_per_buffer = (games//buffers//2)*(2+1+banks)
total_agents = rows_per_buffer*buffers
frozen_rows = historical//buffers  # per buffer, per bank
minibatch = max(horizon, (total_agents*horizon)//4//horizon*horizon)
args = pufferl.load_config('bopl')
args.update(seed=options.seed, reset_state=False, cudagraphs=options.graphs)
args['vec'].update(total_agents=total_agents, num_buffers=buffers, num_threads=options.threads,
                   num_frozen_banks=banks, frozen_bank_pct=frozen_rows/rows_per_buffer+1e-6)
args['policy'].update(hidden_size=options.hidden, num_layers=1)
args['train'].update(total_timesteps=(options.updates+1)*total_agents*horizon, horizon=horizon,
                     minibatch_size=minibatch, learning_rate=.003, ent_coef=.01,
                     anneal_ent_coef=1, min_ent_coef_ratio=.05)
args['env'].update(players=2, banks=banks, historical_every=2, seed=options.seed, action_repeat=4,
                   observation_delay=9, map=0, training_map_mask=1, max_episode_ticks=options.episode_ticks,
                   warmup_ticks=options.warmup_ticks, reward_kill_milli=0, reward_timeout_milli=-500)
pufferl.validate_config(args)
(output/'native-config.json').write_text(json.dumps(args, indent=2)+'\n')

report = dict(schema=1, status='failed', checks=[], records=[], config=dict(vars(options), output=str(output),
              game=os.environ['BOPL_RL_GAME'], total_agents=total_agents))
def check(name, ok, **info):
    report['checks'].append(dict(name=name, ok=bool(ok), **info))
    print(('PASS ' if ok else 'FAIL ')+name, info if not ok else '', flush=True)
    if not ok: raise AssertionError(name)

def active_rows(e):
    return e['slots'] if not e['banks'] else [e['slots'][0], e['slots'][1+e['bank']]]
def idle_rows(e):
    return [] if not e['banks'] else [e['slots'][1+k] for k in range(e['banks']) if k != e['bank']]
def pair_ok(obs, a, b):
    """Two different players who see each other (check_routing's invariant)."""
    ra, rb = obs[a], obs[b]
    if not (ra[PLAYER0] == 1 and ra[PLAYER0+1] == 1 and rb[PLAYER0] == 1 and rb[PLAYER0+1] == 1): return False
    oa = ra[PLAYER0+PLAYER:PLAYER0+2*PLAYER]; ob = rb[PLAYER0+PLAYER:PLAYER0+2*PLAYER]
    if not (oa[0] == 1 and oa[1] == 0 and ob[0] == 1 and ob[1] == 0): return False
    return abs(oa[9]+ob[9]) < 1e-5 and abs(oa[10]+ob[10]) < 1e-5

trainer = None
started = time.monotonic()
try:
    trainer = _C.create_pufferl(args)
    if trainer is None: raise RuntimeError('create_pufferl failed')
    handles = (C.c_void_p*8)()
    check('one native vector', lib.bopl_vec_handles(handles, 8) == 1)
    vec = SimpleNamespace(gpu_obs_ptr=handles[0])
    check('environment count', _C.num_envs(trainer) == games, envs=_C.num_envs(trainer))
    envs = layout(lib, vec, games)
    perm, groups = build_perm(envs, buffers, banks)
    check('bank layout per buffer', all(g == dict(primary=rows_per_buffer-banks*frozen_rows, bank=[frozen_rows]*banks) for g in groups), groups=groups)
    _C.set_agent_perm(trainer, np.array(perm, dtype=np.int32))
    _C.set_env_tags(trainer, np.array([e['tag'] for e in envs], dtype=np.int32))
    config = dict(training_maps=[0], episode_ticks=options.episode_ticks, observation_delay_ticks=9)
    def save(name, count):
        path = output/name
        return path, write_checkpoint(path, lambda p: _C.save_weights(trainer, p), obs_size=OBS, hidden=options.hidden,
                                      num_layers=1, architecture=ARCHITECTURE, game_steps=count['game_steps'],
                                      learner_steps=count['learner_steps'], config=config,
                                      aim_fourier=options.aim_fourier, aim_ar=options.aim_ar)
    initial, initial_meta = save('initial.bin', empty_counters())
    for bank in range(banks): _C.load_frozen_bank(trainer, bank, str(initial))
    check('vector reset after permutation', lib.bopl_vec_reset(C.c_void_p(vec.gpu_obs_ptr)) == 0)
    _C.log(trainer)  # discard warmup/capture loss and profile accumulators
    envs = layout(lib, vec, games)
    expected_slots = []; slot = 0
    for e in envs: expected_slots.append(perm[slot:slot+e['num_agents']]); slot += e['num_agents']
    check('slots follow the permutation', [e['slots'] for e in envs] == expected_slots)
    host = [C.c_void_p(), C.c_void_p(), C.c_void_p()]
    check('host rows exposed', lib.bopl_vec_host(C.c_void_p(vec.gpu_obs_ptr), *[C.byref(h) for h in host]) == total_agents)
    obs = np.ctypeslib.as_array((C.c_float*(total_agents*ROW)).from_address(host[0].value)).reshape(total_agents, ROW)
    terminals = np.ctypeslib.as_array((C.c_float*total_agents).from_address(host[2].value))
    check('reset flags every row for a fresh recurrent state', bool((terminals == 1).all()))
    def rows_ok(label):
        current = layout(lib, vec, games)
        check(f'{label}: active rows are two mutually visible players', all(pair_ok(obs, *active_rows(e)) for e in current))
        stale = [(i, r, int(np.count_nonzero(obs[r]))) for i, e in enumerate(current) for r in idle_rows(e) if obs[r].any()]
        check(f'{label}: idle bank rows are blank', not stale, stale=stale[:16], stale_count=len(stale))
        check(f'{label}: observations finite', bool(np.isfinite(obs).all()))
        return current
    rows_ok('after reset')
    previous = counters.snapshot(lib, vec)
    check('counters start at zero', previous['game_steps'] == previous['learner_steps'] == previous['matches'] == 0, counters=previous)
    report['setup_seconds'] = time.monotonic()-started
    tick = time.monotonic(); previous_elapsed = 0.; loop_seconds = 0.
    def one_update(number, pool_steps):
        global previous, previous_elapsed, loop_seconds
        t0 = time.monotonic(); _C.rollouts(trainer); t1 = time.monotonic(); _C.train(trainer); t2 = time.monotonic()
        loop_seconds += t2-t0
        native = _C.log(trainer)
        count = counters.snapshot(lib, vec)
        elapsed = time.monotonic()-tick
        record = metrics_update(native, count, previous, elapsed=elapsed, previous_elapsed=previous_elapsed,
                                entropy_coefficient=native['loss']['entropy_coefficient'], learning_rate=native['loss']['learning_rate'],
                                pool_steps=pool_steps, rollout_seconds=t1-t0, train_seconds=t2-t1)
        env = native['env']
        check(f'update {number}: no game errors or observation overflow', env.get('game_errors', 0) == 0 and env.get('observation_overflow', 0) == 0, env=env)
        check(f'update {number}: physical decisions', count['game_steps'] == number*games*2*horizon, counters=count)
        check(f'update {number}: learner decisions exclude frozen rows', count['learner_steps'] == number*(games+historical)*horizon, counters=count)
        check(f'update {number}: opponent decisions', count['opponent_game_decisions'] == number*historical*horizon, counters=count)
        check(f'update {number}: padded rows', native['agent_steps'] == number*total_agents*horizon, agent_steps=native['agent_steps'])
        heads = ('entropy_aim', 'entropy_buttons', 'entropy_hold')  # one per action head (layout.ACTION_SIZES)
        for field in heads + ('advantage_std', 'gradient_norm', 'policy_loss', 'value_loss', 'kl'):
            check(f'update {number}: finite {field}', record.get(field) is not None and math.isfinite(record[field]), value=record.get(field))
        check(f'update {number}: joint entropy is the sum of the heads', abs(record['entropy']-sum(record[h] for h in heads)) < 2e-5, record=record)
        check(f'update {number}: kill shaping off', record['behavior']['totals']['reward_kills'] == record['behavior']['totals']['reward_deaths'] == 0)
        rows_ok(f'update {number}')
        report['records'].append(record)
        previous = count; previous_elapsed = elapsed
        print(json.dumps(dict(update=number, learner_steps=count['learner_steps'], game_steps=count['game_steps'], matches=count['matches'],
                              entropy=record['entropy'], learner_sps=round(record['learner_sps']))), flush=True)
        return count
    for number in range(1, options.updates+1): count = one_update(number, [0])
    check('matches completed', count['matches'] > 0, matches=count['matches'])
    final, final_meta = save('final.bin', count)
    check('weights changed', initial_meta['weights_sha256'] != final_meta['weights_sha256'])
    # GPU round trip: reload the checkpoint into the trainer and save again.
    _C.load_weights(trainer, str(final)); again = output/'again.bin'; _C.save_weights(trainer, str(again))
    check('GPU reload preserves weights', again.read_bytes() == final.read_bytes())
    policy, restored = load_policy(final)
    check('CPU evaluator reload', restored['step'] == count['game_steps'] and restored['learner_steps'] == count['learner_steps'], meta=restored)
    check('CPU evaluator step is finite', all(torch.isfinite(v).all() for v in policy.step(torch.zeros(2, OBS), policy.initial(2))))
    check('CPU evaluator aim parameterisation', (policy.aim_fourier, policy.aim_ar) == (options.aim_fourier, options.aim_ar)
          and policy.width == policy_layout.decoder_logits(options.aim_fourier, options.aim_ar), width=policy.width)
    # Pool rotation: bank 0 now serves the trained weights, bank 1 keeps the initial ones.
    _C.load_frozen_bank(trainer, 0, str(final))
    count = one_update(options.updates+1, [0, count['game_steps']])
    check('rotation keeps games running', count['matches'] >= previous['matches'], matches=count['matches'])
    learner_decisions = count['learner_steps']
    report.update(status='passed', checkpoint=str(final), counters=count, native_agent_steps=trainer.global_step,
                  loop_seconds=loop_seconds, learner_dps_loop=learner_decisions/loop_seconds,
                  learner_dps_active=learner_decisions/(time.monotonic()-tick))
    print(json.dumps(dict(status='passed', learner_decisions=learner_decisions, loop_seconds=round(loop_seconds, 2),
                          learner_dps_loop=round(report['learner_dps_loop']), learner_dps_active=round(report['learner_dps_active']),
                          setup_seconds=round(report['setup_seconds'], 2))), flush=True)
finally:
    if trainer is not None: _C.close(trainer)
    report['elapsed_seconds'] = time.monotonic()-started
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
