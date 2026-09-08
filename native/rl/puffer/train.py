"""Train native recurrent PPO with match-stable historical opponents."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]
from common import atomic_json
from native_checkpoint import write_checkpoint
from native_metrics import empty_counters, update
from native_pool import SnapshotPool
from native_vector import NativeVector
import reference


def maps(value):
    try:
        result = [int(x) for x in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected comma-separated map IDs') from error
    if not result or len(set(result)) != len(result) or any(x < 0 or x > 30 for x in result):
        raise argparse.ArgumentTypeError('Expected distinct map IDs in 0..30')
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--game', type=Path)
    p.add_argument('--steps', type=int, default=100_000, help='Actual learner decisions (excludes frozen players)')
    p.add_argument('--workers', type=int, default=16, help='Number of simultaneous two-player games')
    p.add_argument('--buffers', type=int, choices=(1, 2))
    p.add_argument('--banks', type=int, choices=range(2, 9), default=4)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--learner-passes', type=int, default=1, help='Native minibatches per rollout; each has one learner batch of samples')
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--entropy', type=float, default=.01)
    p.add_argument('--entropy-floor', type=float, default=.0005)
    p.add_argument('--episode-ticks', type=int, default=7200)
    p.add_argument('--training-maps', type=maps, default=[0])
    p.add_argument('--checkpoint-every', type=int, default=1_000_000, help='Learner decisions between immutable snapshots')
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
    a = p.parse_args()
    for name in ('steps', 'workers', 'horizon', 'layers', 'learner_passes', 'episode_ticks',
                 'checkpoint_every', 'eval_every', 'evaluation_workers'):
        if getattr(a, name) < 1:
            p.error(f'{name.replace("_", "-")} must be positive')
    a.buffers = a.buffers or (2 if a.workers % 4 == 0 else 1)
    if a.workers % (2 * a.buffers):
        p.error('workers must be divisible by twice the buffer count')
    if a.hidden < 8 or a.hidden % 8:
        p.error('hidden must be a positive multiple of eight')
    if a.cudagraphs != -1 and a.cudagraphs < 1:
        p.error('cudagraphs must be -1 or at least 1')
    if not (math.isfinite(a.learning_rate) and a.learning_rate > 0
            and 0 < a.entropy_floor <= a.entropy and math.isfinite(a.entropy)):
        p.error('Expected positive learning rate and 0 < entropy-floor <= entropy')
    a.evaluation_maps = a.evaluation_maps or a.training_maps
    a.heldout_maps = a.heldout_maps or [m for m in range(31) if m not in a.training_maps][:2]
    if not a.no_eval:
        if not a.heldout_maps or set(a.heldout_maps) & set(a.training_maps + a.evaluation_maps):
            p.error('heldout maps must be disjoint from training and evaluation maps')
        if any(n < 2 or n % 2 for n in (a.evaluation_games, a.heldout_games, a.frozen_games)):
            p.error('evaluation game counts must be positive and even')
    return a


def main():
    a = arguments()
    output = a.output.resolve()
    game = reference.resolve_game(a.game)
    runtime = {key: hashlib.sha256(path.read_bytes()).hexdigest()
               for key, path in reference.player_sources(game).items()}
    output.mkdir(parents=True, exist_ok=False)
    for key, value in dict(BOPL_RL_GAME=game, BOPL_RL_BOOTSTRAP=ROOT/'native/fixtures/movement.rep',
                           BOPL_RL_LOG_DIR=output/'game-logs', OMP_NUM_THREADS=a.workers).items():
        os.environ[key] = str(value)
    (output/'game-logs').mkdir()
    import numpy as np
    import torch
    from pufferlib import _C, pufferl
    if not _C.gpu:
        raise RuntimeError('Native training requires the CUDA build')
    torch.cuda.init()
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    # PufferLib parses sys.argv itself; the wrapper owns this command's arguments.
    sys.argv = [sys.argv[0]]
    args = pufferl.load_config('bopl')
    vector = NativeVector(_C, a.workers, a.buffers, a.banks)
    learner_batch = vector.learner_rows * a.horizon
    native_batch = vector.total_agents * a.horizon
    updates = math.ceil(a.steps / learner_batch)
    # Native replay_ratio multiplies padded rows. Correct for those rows, and
    # round up one float32 ULP because C truncates the resulting update count.
    ratio = float(np.nextafter(np.float32(a.learner_passes * learner_batch / native_batch), np.float32(np.inf)))
    args.update(seed=a.seed, reset_state=False, cudagraphs=a.cudagraphs)
    args['vec'].update(total_agents=vector.total_agents, num_buffers=a.buffers,
                       num_threads=a.workers, num_frozen_banks=a.banks,
                       frozen_bank_pct=(vector.pairs + .000001) / vector.agents_per_buffer)
    args['policy'].update(hidden_size=a.hidden, num_layers=a.layers)
    args['train'].update(total_timesteps=updates*native_batch, seed=a.seed,
                         horizon=a.horizon, minibatch_size=learner_batch, replay_ratio=ratio,
                         learning_rate=a.learning_rate, ent_coef=a.entropy,
                         anneal_ent_coef=1, min_ent_coef_ratio=a.entropy_floor/a.entropy)
    args['env'].update(players=2, banks=a.banks, historical_every=2, seed=a.seed,
                       action_repeat=4, observation_delay=9, map=a.training_maps[0],
                       training_map_mask=sum(1 << m for m in a.training_maps),
                       max_episode_ticks=a.episode_ticks, reward_kill_milli=0, reward_timeout_milli=-500)
    pufferl.validate_config(args)
    target_physical = math.ceil(a.steps * 4 / 3)
    config = dict(schema=1, algorithm='recurrent PPO', backend='pufferlib-native',
                  step_unit='physical_player_decisions', steps=target_physical, learner_steps=a.steps,
                  seed=a.seed, workers=a.workers, banks=a.banks, hidden=a.hidden, layers=a.layers,
                  training_maps=a.training_maps, evaluation_maps=a.evaluation_maps,
                  heldout_maps=a.heldout_maps, episode_ticks=a.episode_ticks,
                  observation_delay_ticks=9, telemetry_schema=1,
                  reward=dict(win=1, loss=-1, timeout=-.5, kill=0, death=0),
                  entropy=a.entropy, entropy_floor=a.entropy_floor, learner_passes=a.learner_passes,
                  frozen_opponents=[], evaluation_baseline=str(a.evaluation_baseline.resolve()) if a.evaluation_baseline else None,
                  game=str(game), runtime=runtime, native=args,
                  native_binding_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest())
    atomic_json(output/'config.json', config)
    started = time.monotonic()
    counts = empty_counters()
    trainer = None
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
    try:
        trainer = _C.create_pufferl(args)
        vector.attach(trainer)
        def save():
            step = counts['game_steps']
            if step not in saved:
                path = output/'checkpoints'/f'policy-{step:09d}.bin'
                write_checkpoint(path, lambda name: _C.save_weights(trainer, name), obs_size=3257,
                                 hidden=a.hidden, num_layers=a.layers, architecture='puffernet-mingru',
                                 game_steps=step, learner_steps=counts['learner_steps'], config=config)
                saved[step] = path
            return saved[step]
        initial = save()
        pool = SnapshotPool(_C, trainer, vector, a.banks, initial)
        vector.reset()
        _C.log(trainer)  # Warmup losses are not measured training updates.
        counts = vector.snapshot()
        if counts['learner_steps'] or counts['game_steps']:
            raise RuntimeError('Setup unexpectedly counted game decisions')
        config['evaluation_baseline'] = str(a.evaluation_baseline.resolve() if a.evaluation_baseline else initial)
        atomic_json(output/'config.json', config)
        evaluated_steps = set()
        def evaluate(checkpoint):
            command = [sys.executable, str(ROOT/'native/rl/research/evaluate.py'), '--run', str(output),
                       '--checkpoint', str(checkpoint), '--initial', config['evaluation_baseline'],
                       '--evaluation-maps', ','.join(map(str, a.evaluation_maps)),
                       '--heldout-maps', ','.join(map(str, a.heldout_maps)),
                       '--games-per-opponent', str(a.evaluation_games), '--heldout-games', str(a.heldout_games),
                       '--frozen-games', str(a.frozen_games), '--workers', str(a.evaluation_workers), '--game', str(game)]
            if a.evaluation_opponents:
                command += ['--opponents', str(a.evaluation_opponents.resolve())]
            with (output/'evaluation.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            evaluated_steps.add(counts['game_steps'])
        next_checkpoint, next_eval = a.checkpoint_every, a.eval_every
        previous, previous_elapsed = counts, 0.
        training_started = time.monotonic()
        with (output/'metrics.jsonl').open('a') as metrics, (output/'pool.jsonl').open('a') as events:
            def event(value):
                if value:
                    events.write(json.dumps(value, allow_nan=False)+'\n'); events.flush()
            while counts['learner_steps'] < a.steps and not stop:
                before = time.monotonic()
                _C.rollouts(trainer)
                rolled = time.monotonic()
                _C.train(trainer)
                trained = time.monotonic()
                native = _C.log(trainer)
                counts = vector.snapshot()
                if (counts['learner_steps']-previous['learner_steps'] != learner_batch
                        or counts['game_steps']-previous['game_steps'] != a.workers*2*a.horizon):
                    raise RuntimeError('Native rollout physical/learner decision accounting mismatch')
                elapsed = time.monotonic()-training_started
                record = update(native, counts, previous, elapsed=elapsed, previous_elapsed=previous_elapsed,
                                entropy_coefficient=native['loss']['entropy_coefficient'],
                                learning_rate=native['loss']['learning_rate'], pool_steps=pool.steps,
                                rollout_seconds=rolled-before, train_seconds=trained-rolled)
                metrics.write(json.dumps(record, allow_nan=False)+'\n'); metrics.flush()
                print(json.dumps(dict(step=record['step'], learner_steps=record['learner_steps'],
                                      learner_sps=record['learner_sps'], matches=record['matches'])), flush=True)
                previous, previous_elapsed = counts, elapsed
                event(pool.service())
                if counts['learner_steps'] >= next_checkpoint:
                    event(pool.request(save(), counts['game_steps']))
                    next_checkpoint = (counts['learner_steps']//a.checkpoint_every+1)*a.checkpoint_every
                status('running')
                if not a.no_eval and counts['learner_steps'] >= next_eval and not stop:
                    evaluate(save())
                    next_eval = (counts['learner_steps']//a.eval_every+1)*a.eval_every
            final = save()
            if not a.no_eval and not stop and counts['game_steps'] not in evaluated_steps:
                evaluate(final)
        status('stopped' if stop else 'complete', checkpoint=str(final))
    except BaseException as error:
        status('failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        if trainer is not None:
            _C.close(trainer)


if __name__ == '__main__':
    main()
