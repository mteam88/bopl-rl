"""Train native recurrent PPO against PufferLib's Elo-rated self-play pool."""
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
from evaluation_protocol import SCRIPTED_KINDS, SCRIPTED_LEVELS, checkpoint_identity, opponents, DEFAULT_EVAL_SCRIPTED
from native_checkpoint import write_checkpoint
import layout
from native_metrics import empty_counters, update
import native_selfplay
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


def scripted_bots(value):
    """Comma-separated bot kinds (bopl_bots.h) that take over the last banks."""
    names = [x for x in value.split(',') if x]
    if any(name not in SCRIPTED_KINDS for name in names):
        raise argparse.ArgumentTypeError(f'Expected bot kinds from {sorted(SCRIPTED_KINDS)}')
    return names


def scripted_share(learner_steps, decay):
    """Fraction of a scripted bank's matches the bot plays: 1 without a decay, else linear to 0 at `decay`."""
    return 1.0 if not decay else max(0.0, 1.0 - learner_steps / decay)


def set_scripted_share(_C, share):
    """Write the C environment's process-wide bot share (bopl_puffer.h bopl_scripted_share)."""
    import ctypes
    ctypes.c_float.in_dll(ctypes.CDLL(_C.__file__), 'bopl_scripted_share').value = float(share)


def entropy_coefficient(learner_steps, initial, floor, half_life):
    """Exponential decay in learner decisions, independent of the run budget."""
    return max(floor, initial * math.exp2(-learner_steps / half_life))


# Resume (--resume): a run directory is restarted from its last checkpoint and
# may be given a larger --steps (every schedule is a function of learner
# decisions or constant, never of the run budget). Restored: policy weights,
# the self-play pool (entries, Elo, per-bank opponents, sampling RNG), the
# frozen banks' weights, the native step counter (pool snapshot names and
# intervals), and the cumulative environment counters, which the C vector
# restarts at zero and this process offsets. Not restored: the optimizer's
# Muon momentum buffers (no native API; they rebuild within a few updates)
# and the recurrent state, which starts as at an episode boundary.
RESUME_HYPERS = ('workers', 'buffers', 'banks', 'seed', 'encoder', 'hidden', 'layers', 'horizon',
                 'learner_passes', 'learning_rate', 'anneal_lr', 'entropy', 'entropy_floor',
                 'entropy_half_life', 'episode_ticks', 'stocks', 'training_maps', 'reward_kill',
                 'reward_timeout', 'pool_size', 'swap_winrate', 'min_games', 'snapshot_interval',
                 'opponent_timeout', 'elo_init', 'elo_k', 'pool_seed', 'head_scale', 'entropy_free_rows',
                 'scripted_bots', 'scripted_level', 'scripted_decay', 'aim_fourier', 'aim_ar')
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
                reward_timeout=config['reward']['timeout'], pool_size=pool['max_size'], swap_winrate=pool['swap_winrate'],
                min_games=pool['min_games'], snapshot_interval=pool['snapshot_interval'],
                opponent_timeout=pool['opponent_timeout'], elo_init=pool['elo_init'], elo_k=pool['elo_k'],
                pool_seed=pool['seed'], head_scale=config['entropy_head_scale'],
                entropy_free_rows=config.get('entropy_free_rows', 0),
                scripted_bots=config.get('scripted_bots', []), scripted_level=config.get('scripted_level', 'hard'),
                scripted_decay=config.get('scripted_decay', 0),
                aim_fourier=config.get('aim_fourier', 0), aim_ar=config.get('aim_ar', 0))


def offset_counters(counts, base):
    """Cumulative counters of this process plus those of the runs before it."""
    if base is None:
        return counts
    out = dict(counts)
    for key in COUNTER_KEYS:
        out[key] = counts[key] + base[key]
    out['buttons'] = [x + y for x, y in zip(counts['buttons'], base['buttons'])]
    out['telemetry'] = [x + y for x, y in zip(counts['telemetry'], base['telemetry'])]
    out['map_matches'] = {m: counts['map_matches'].get(m, 0) + base['map_matches'].get(m, 0)
                          for m in set(counts['map_matches']) | set(base['map_matches'])}
    return out


def pool_state_json(pool, ledger, *, counts, elapsed, global_step, update_index, checkpoint):
    return dict(schema=1, pool={k: v for k, v in pool.items() if k != 'rng'},
                rng=pool['rng'].bit_generator.state, ledger=ledger, counts=counts, elapsed=elapsed,
                global_step=int(global_step), update_index=update_index, checkpoint=str(checkpoint))


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
    p.add_argument('--game', type=Path)
    p.add_argument('--steps', type=int, default=100_000, help='Actual learner decisions (excludes frozen players)')
    p.add_argument('--workers', type=int, default=16, help='Number of simultaneous two-player games')
    p.add_argument('--buffers', type=int, choices=(1, 2, 4, 8),
                   help='Rollout buffers (independent buffer threads + CUDA streams over one env pool). 4 measured +11%% learner/s over 2 '
                        'at 128 workers on the f64 sim (2026-09-10): GPU sync of one buffer overlaps env stepping of the others')
    p.add_argument('--banks', type=int, choices=range(2, 9), default=4)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--encoder', choices=('linear', 'set'), default='linear')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--learner-passes', type=int, default=1, help='Native minibatches per rollout; each has one learner batch of samples')
    p.add_argument('--minibatch-div', type=int, default=1,
                   help='Split each learner batch into this many minibatches (optimizer steps); must divide the learner rows. '
                        'Throughput knob: smaller minibatches give the 16-SM training partition less work per GEMM but more launches; default 1')
    p.add_argument('--learning-rate', type=float, default=.0003)
    p.add_argument('--anneal-lr', type=int, choices=(0, 1), default=0,
                   help='0 keeps learning rate constant; 1 uses native cosine decay over the run budget')
    p.add_argument('--min-lr-ratio', type=float, default=0.,
                   help='Final/base learning-rate ratio when --anneal-lr 1')
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
                        'harmonics per aim block instead of 256 bins (puffer/bopl_aim.h); 0 keeps raw bins')
    p.add_argument('--aim-ar', type=int, choices=(0, 1), default=0,
                   help='1 conditions the aim head on the sampled button combo: one aim block per combo, buttons '
                        'sampled before aim (in the trainer, the CPU evaluator and the in-game agent)')
    p.add_argument('--episode-ticks', type=int, default=7200)
    p.add_argument('--reward-kill', type=float, default=0.,
                   help='Per-kill bonus, also paid as a per-death penalty (game credit incl. self-kills)')
    p.add_argument('--reward-timeout', type=float, default=-.5,
                   help='Terminal reward for both players when the episode hits --episode-ticks (<= 0)')
    p.add_argument('--stocks', type=int, default=0, help='Smash-style extra-life setting; 0 = vanilla Bopl, one life per round')
    p.add_argument('--training-maps', type=maps, default=[0])
    p.add_argument('--checkpoint-every', type=int, default=1_000_000, help='Learner decisions between immutable snapshots')
    p.add_argument('--pool-size', type=int, default=16, help='Historical snapshots kept by the self-play pool')
    p.add_argument('--swap-winrate', type=float, default=.6,
                   help='Learner win rate against a bank opponent that retires it (0.5 is parity in this symmetric game)')
    p.add_argument('--min-games', type=int, default=256, help='Completed games per bank before its win rate can retire the opponent')
    p.add_argument('--snapshot-interval', type=int, default=10_000_000, help='Learner decisions between pool snapshots taken regardless of win rate')
    p.add_argument('--opponent-timeout', type=int, default=50_000_000, help='Learner decisions a bank keeps one opponent before a forced swap')
    p.add_argument('--elo-init', type=float, default=0.)
    p.add_argument('--elo-k', type=float, default=16.)
    p.add_argument('--pool-seed', type=int, help='Opponent sampling seed (defaults to --seed)')
    p.add_argument('--scripted-bots', type=scripted_bots, default='',
                   help='Comma-separated bot kinds (archer,pursuer,evasive,mixed) that permanently take over the '
                        'last banks: those banks play the scripted bot instead of frozen snapshots, so the fraction '
                        'of the learner\'s historical games against bots is len(list)/banks')
    p.add_argument('--scripted-level', choices=sorted(SCRIPTED_LEVELS, key=SCRIPTED_LEVELS.get), default='hard')
    p.add_argument('--scripted-decay', type=int, default=0,
                   help='Learner decisions over which the scripted banks hand their matches to self-play: the bot '
                        'share falls linearly from 1 to 0 at this step (0 = bots play every match of their banks). '
                        'With a decay the pool manages the scripted banks too, so their frozen weights are real opponents')
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
    a = p.parse_args()
    for name in ('steps', 'workers', 'horizon', 'layers', 'learner_passes', 'episode_ticks',
                 'checkpoint_every', 'eval_every', 'evaluation_workers', 'entropy_half_life',
                 'pool_size', 'min_games', 'snapshot_interval', 'opponent_timeout'):
        if getattr(a, name) < 1:
            p.error(f'{name.replace("_", "-")} must be positive')
    if not 0 < a.swap_winrate <= 1 or not math.isfinite(a.elo_init) or not math.isfinite(a.elo_k) or a.elo_k <= 0:
        p.error('Expected 0 < swap-winrate <= 1, finite elo-init and positive elo-k')
    a.pool_seed = a.seed if a.pool_seed is None else a.pool_seed
    a.buffers = a.buffers or (2 if a.workers % 4 == 0 else 1)
    if a.workers % (2 * a.buffers):
        p.error('workers must be divisible by twice the buffer count')
    if (a.workers // 2) % a.banks:
        p.error('half the workers must divide evenly among the banks')
    if len(a.scripted_bots) >= a.banks:
        p.error('scripted-bots must leave at least one policy bank for the self-play pool')
    if a.scripted_decay < 0 or (a.scripted_decay and not a.scripted_bots):
        p.error('scripted-decay needs scripted-bots and a non-negative step count')
    if a.hidden < 8 or a.hidden % 8:
        p.error('hidden must be a positive multiple of eight')
    if a.cudagraphs != -1 and a.cudagraphs < 1:
        p.error('cudagraphs must be -1 or at least 1')
    if not (math.isfinite(a.learning_rate) and a.learning_rate > 0
            and 0 < a.entropy_floor <= a.entropy and math.isfinite(a.entropy)):
        p.error('Expected positive learning rate and 0 < entropy-floor <= entropy')
    if not math.isfinite(a.min_lr_ratio) or not 0 <= a.min_lr_ratio <= 1:
        p.error('min-lr-ratio must be finite and within 0..1')
    try:
        head_scale = [float(part) for part in a.entropy_head_scale.split(',')]
    except ValueError:
        head_scale = []
    if len(head_scale) != layout.NUM_ATNS or not all(math.isfinite(s) and 0 <= s <= 100 for s in head_scale):
        p.error(f'entropy-head-scale must be {layout.NUM_ATNS} finite values in 0..100')
    a.head_scale = head_scale
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
    frozen = opponents(a.evaluation_opponents)
    baseline_identity = checkpoint_identity(a.evaluation_baseline) if a.evaluation_baseline else None
    # BOPL_RL_BACKEND=rust rolls out on the Rust sim (native/src/rollout,
    # built by `cargo build --release`); anything else keeps the .NET/Unity
    # game resolution. The Rust backend needs no spare world (resets are
    # microseconds) and hosts every world in one runtime.
    if os.environ.get('BOPL_RL_BACKEND', '').lower() == 'rust':
        game = (ROOT/'native/target/release/libbopl_sim.so').resolve()
        if not game.exists():
            raise SystemExit(f'BOPL_RL_BACKEND=rust: {game} is missing; run cargo build --release in native/')
        os.environ.setdefault('BOPL_RL_INPROC_SPARE', '0')
        os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', str(1 << 20))
    else:
        game = reference.resolve_game(a.game)
    # The NativeAOT sim (sim/RL.md) is one self-contained binary: only the
    # player_binary entry exists for it, the Unity-specific files do not.
    runtime = {key: hashlib.sha256(path.read_bytes()).hexdigest()
               for key, path in reference.player_sources(game).items() if path.exists()}
    resume = None
    if a.resume:
        resume = json.loads((output/'pool/state.json').read_text())
        previous_config = json.loads((output/'config.json').read_text())
        recorded = recorded_hypers(previous_config)
        for name in RESUME_HYPERS:
            if recorded[name] != getattr(a, name):
                raise SystemExit(f'--resume: {name} was {recorded[name]!r}, now {getattr(a, name)!r}; only --steps may change')
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
    # SIM_OVERHAUL.md section 4: one GC heap under every world costs 25-45%
    # of throughput (each gen0 collection suspends every world); four worlds
    # per runtime measured within 5-15% of one runtime per world at a
    # fraction of the memory. Honour an explicit setting.
    os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', '4')
    # The rollout steps games on vecenv.h's own env pool (THROUGHPUT.md 3.D), not
    # OpenMP: BOPL_ENV_THREADS stepping threads in total (buffer threads
    # included; default CPUs - 3, i.e. 9 on the 12-thread box: 8-10 tie at
    # 128 workers, 10 is +3% at 256, 12+ oversubscribes and stalls on preempted
    # workers), BOPL_ENV_LPT=1 slowest-first claim order, BOPL_ENV_STEAL=1
    # buffer threads step other buffers' games while their own forward runs,
    # BOPL_ENV_SPIN_US=100 idle spin before a futex park, BOPL_GPU_WAIT=poll|sync,
    # BOPL_ENV_PIN=1 to pin threads, BOPL_ENV_PROF=1|2 pool statistics and the
    # c_step phase attribution on stderr.
    # BOPL_ASYNC=1 (default) overlaps each rollout with the previous batch's
    # training (see the update loop); 0 is the serial trainer. With overlap
    # BOPL_GPU_SPLIT=<SMs for inference>|auto|0 partitions the GPU between the
    # rollout and train streams (green contexts; auto = one half, 24/24 on
    # the 48-SM 5070; 0 shares the SMs on stream priority alone) and
    # BOPL_TRAIN_WAIT=block|spin picks how the trainer waits for its stream.
    # Rollout forward (THROUGHPUT.md 3.C, bopl_rollout.cu): BOPL_BATCHED_FWD=1
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
    os.environ['BOPL_ENT_HEAD_SCALE'] = ','.join(repr(s) for s in a.head_scale)
    os.environ['BOPL_AIM_FOURIER'] = str(a.aim_fourier)  # read once by the trainer (bopl_aim_config)
    os.environ['BOPL_AIM_AR'] = str(a.aim_ar)
    import numpy as np
    import torch
    from pufferlib import _C, pufferl, selfplay
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
    # trainer's bank_layout ends before the idle tail (THROUGHPUT.md 3.C).
    args['vec'].update(total_agents=vector.total_agents, num_buffers=a.buffers,
                       num_threads=a.workers, num_frozen_banks=a.banks,
                       frozen_bank_pct=(vector.live_per_bank + .000001) / vector.agents_per_buffer,
                       primary_rows=vector.primary_rows)
    args['policy'].update(hidden_size=a.hidden, num_layers=a.layers)
    args['train'].update(total_timesteps=updates*native_batch, seed=a.seed,
                         horizon=a.horizon, minibatch_size=minibatch, replay_ratio=ratio,
                         learning_rate=a.learning_rate, anneal_lr=a.anneal_lr, min_lr_ratio=a.min_lr_ratio,
                         ent_coef=a.entropy, anneal_ent_coef=0,
                         min_ent_coef_ratio=a.entropy_floor/a.entropy)
    # Scripted banks are the LAST ones so the self-play pool, which only
    # manages banks 0..policy_banks-1, never swaps weights into them. Under a
    # decay the bots hand a growing share of their banks' matches to the banks'
    # frozen weights, so the pool must manage every bank; bot matches stay out
    # of the pool's statistics (bopl_puffer.h bot_*_bank).
    policy_banks = a.banks if a.scripted_decay else a.banks - len(a.scripted_bots)
    scripted_first_bank = a.banks - len(a.scripted_bots)
    scripted_kinds = 0
    for i, name in enumerate(a.scripted_bots):
        scripted_kinds |= SCRIPTED_KINDS[name] << 4 * (scripted_first_bank + i)
    args['env'].update(players=2, banks=a.banks, historical_every=2, seed=a.seed,
                       action_repeat=4, observation_delay=9, map=a.training_maps[0],
                       training_map_mask=sum(1 << m for m in a.training_maps),
                       max_episode_ticks=a.episode_ticks, stocks=a.stocks, reward_kill_milli=round(a.reward_kill*1000), reward_timeout_milli=round(a.reward_timeout*1000),
                       scripted_kinds=scripted_kinds, scripted_level=SCRIPTED_LEVELS[a.scripted_level])
    pufferl.validate_config(args)
    target_physical = math.ceil(a.steps * 4 / 3)
    # PufferLib's pool counts native agent steps (padded rows included);
    # the knobs above count learner decisions.
    native_per_learner = native_batch / learner_batch
    pool_config = dict(backend='pufferlib.selfplay', banks=policy_banks, historical_games_per_bank=vector.historical_per_bank,
                    max_size=a.pool_size, swap_winrate=a.swap_winrate, min_games=a.min_games,
                    snapshot_interval=a.snapshot_interval, opponent_timeout=a.opponent_timeout,
                    elo_init=a.elo_init, elo_k=a.elo_k, seed=a.pool_seed, step_unit='learner_decisions',
                    native_steps_per_learner_decision=native_per_learner, mid_match_swaps='counted in neither bank statistic')
    config = dict(schema=1, algorithm='recurrent PPO', backend='pufferlib-native',
                  step_unit='physical_player_decisions', steps=target_physical, learner_steps=a.steps,
                  seed=a.seed, workers=a.workers, banks=a.banks, encoder=a.encoder, hidden=a.hidden, layers=a.layers,
                  scripted_bots=a.scripted_bots, scripted_level=a.scripted_level, scripted_decay=a.scripted_decay,
                  training_maps=a.training_maps, evaluation_maps=a.evaluation_maps,
                  heldout_maps=a.heldout_maps, episode_ticks=a.episode_ticks, stocks=a.stocks,
                  observation_delay_ticks=9, telemetry_schema=1,
                  reward=dict(win=1, loss=-1, timeout=a.reward_timeout, kill=a.reward_kill, death=-a.reward_kill),
                  entropy=a.entropy, entropy_floor=a.entropy_floor, entropy_head_scale=a.head_scale,
                  entropy_free_rows=a.entropy_free_rows,
                  aim_fourier=a.aim_fourier, aim_ar=a.aim_ar,
                  entropy_schedule='exponential', entropy_half_life=a.entropy_half_life,
                  layout=dict(version=2, obs_size=layout.OBS_SIZE, action_sizes=layout.ACTION_SIZES,
                              hold_lengths=layout.HOLD_LENGTHS),
                  learning_rate_schedule='cosine' if a.anneal_lr else 'constant', learner_passes=a.learner_passes,
                  selfplay=pool_config,
                  frozen_opponents=frozen, evaluation_baseline_identity=baseline_identity, evaluation_baseline=str(a.evaluation_baseline.resolve()) if a.evaluation_baseline else None,
                  game=str(game), runtime=runtime, native=args,
                  native_binding_sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest())
    if resume:
        # The original config stays the record of the run; only the budget and
        # the resume history change.
        config = previous_config
        config.update(steps=target_physical, learner_steps=a.steps)
        config.setdefault('resumed', []).append(dict(at_learner_steps=resume['counts']['learner_steps'],
                                                     checkpoint=resume['checkpoint'], steps=a.steps))
    atomic_json(output/'config.json', config)
    started = time.monotonic()
    counts = offset_counters(empty_counters(), resume['counts'] if resume else None)
    trainer = None
    pending = []  # background evaluation (process, log) pairs
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
        pool = ledger = None
        update_index = 0
        elapsed_before = 0.
        def save():
            step = counts['game_steps']
            if step not in saved:
                path = output/'checkpoints'/f'policy-{step:09d}.bin'
                write_checkpoint(path, lambda name: _C.save_weights(trainer, name), obs_size=layout.OBS_SIZE,
                                 hidden=a.hidden, num_layers=a.layers, architecture=layout.ARCHITECTURE if a.encoder == 'set' else 'puffernet-mingru',
                                 game_steps=step, learner_steps=counts['learner_steps'], config=config,
                                 aim_fourier=a.aim_fourier, aim_ar=a.aim_ar)
                saved[step] = path
            if pool is not None:
                # Everything --resume needs besides the weights, written with
                # every checkpoint so the last one is always consistent.
                atomic_json(output/'pool/state.json', pool_state_json(
                    pool, ledger, counts=counts, elapsed=elapsed_before + time.monotonic() - training_started,
                    global_step=trainer.global_step, update_index=update_index, checkpoint=saved[step]))
            return saved[step]
        training_started = time.monotonic()
        if resume:
            saved = {int(path.stem.split('-')[1]): path for path in sorted((output/'checkpoints').glob('policy-*.bin'))}
            initial = saved[min(saved)]
            _C.load_weights(trainer, resume['checkpoint'])
            trainer.global_step = resume['global_step']
            pool, ledger = load_pool_state(resume)
            if len(pool['banks']) != policy_banks:
                raise SystemExit(f"--resume: the pool manages {len(pool['banks'])} banks but "
                                 f'--banks/--scripted-bots now leave {policy_banks} policy banks')
            for bank, state in enumerate(pool['banks']):
                _C.load_frozen_bank(trainer, bank, state['cur_opp_path'])
            # Scripted banks never swap: their GPU rows run the initial policy
            # (actions unread; the C environment drives those players).
            for bank in range(policy_banks, a.banks):
                _C.load_frozen_bank(trainer, bank, str(initial))
            update_index = resume['update_index']
            elapsed_before = resume['elapsed']
        else:
            initial = save()
            # Every bank starts from the initial policy; PufferLib's pool swaps
            # them independently as the learner outgrows each opponent.
            for bank in range(a.banks):
                _C.load_frozen_bank(trainer, bank, str(initial))
            (output/'pool').mkdir()
            pool = native_selfplay.state(output/'pool', str(initial), policy_banks, vector.historical_per_bank, trainer.global_step,
                                         max_size=a.pool_size, swap_winrate=a.swap_winrate, min_games=a.min_games,
                                         snapshot_interval=round(a.snapshot_interval*native_per_learner),
                                         opp_timeout_steps=round(a.opponent_timeout*native_per_learner),
                                         elo_init=a.elo_init, elo_k=a.elo_k, seed=a.pool_seed)
            ledger = native_selfplay.members(str(initial))
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
            command = [sys.executable, str(ROOT/'native/rl/research/evaluate.py'), '--run', str(output),
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
        next_checkpoint = (counts['learner_steps']//a.checkpoint_every+1)*a.checkpoint_every
        next_eval = (counts['learner_steps']//a.eval_every+1)*a.eval_every
        previous, previous_elapsed = counts, elapsed_before
        free, applied_scale = None, list(a.head_scale)
        share = scripted_share(counts['learner_steps'], a.scripted_decay)
        if a.scripted_bots:
            set_scripted_share(_C, share)
        # Overlap (THROUGHPUT.md 3.A, BOPL_ASYNC=1 default; 0 restores the
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
        if overlap:
            rollout_seconds = _C.rollouts(trainer)  # bootstrap: the first slot before the loop
            ready = offset_counters(vector.snapshot(), base_counts)
        training_started = time.monotonic()
        with (output/'metrics.jsonl').open('a') as metrics, (output/'pool.jsonl').open('a') as events:
            while counts['learner_steps'] < a.steps and not stop:
                before = time.monotonic()
                if overlap:
                    # Another update follows this one: collect its batch now.
                    more = ready['learner_steps'] < a.steps
                    if a.scripted_decay:
                        # Matches starting during the next rollout draw against this share.
                        share = scripted_share(counts['learner_steps'], a.scripted_decay)
                        set_scripted_share(_C, share)
                    if more:
                        _C.rollouts_start(trainer)
                else:
                    rollout_seconds = _C.rollouts(trainer)
                    if a.scripted_decay:
                        share = scripted_share(counts['learner_steps'], a.scripted_decay)
                        set_scripted_share(_C, share)
                rolled = time.monotonic()
                # Configure the native loss before training. Native train_impl
                # copies this scalar into a device buffer each update, so CUDA
                # graph replay observes the new coefficient. Count only the
                # learner decisions completed before this rollout; warmup and
                # frozen/padded rows never advance the schedule.
                trainer.hypers.ent_coef = entropy_coefficient(
                    counts['learner_steps'], a.entropy, a.entropy_floor, a.entropy_half_life)
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
                free = free_fraction(native['env']) if a.entropy_free_rows else None
                if overlap:
                    counts = ready
                    if more:
                        ready = offset_counters(vector.snapshot(), base_counts)
                else:
                    counts = offset_counters(vector.snapshot(), base_counts)
                if (counts['learner_steps']-previous['learner_steps'] != learner_batch
                        or counts['game_steps']-previous['game_steps'] != a.workers*2*a.horizon):
                    raise RuntimeError('Native rollout physical/learner decision accounting mismatch')
                elapsed = elapsed_before + time.monotonic()-training_started
                # Elo, snapshots and bank swaps from this rollout's finished games.
                before_pool = native_selfplay.summary(pool, ledger)
                flat = {'env/'+key: value for key, value in native['env'].items()}
                native_selfplay.credit(pool, flat, ledger)
                native_selfplay.step(selfplay.step, trainer, _C, pool, flat, update_index)
                update_index += 1
                native_selfplay.register(pool, ledger, counts['game_steps'], counts['learner_steps'])
                after_pool = native_selfplay.summary(pool, ledger)
                record = update(native, counts, previous, elapsed=elapsed, previous_elapsed=previous_elapsed,
                                entropy_coefficient=native['loss']['entropy_coefficient'],
                                learning_rate=native['loss']['learning_rate'],
                                pool_steps=sorted(bank['opponent_step'] for bank in after_pool['banks']),
                                rollout_seconds=rollout_seconds, train_seconds=trained-rolled)
                # With overlap the rollout (timed on the buffer threads) and the
                # train phase run concurrently, so they no longer sum to the
                # update's wall time; wall_seconds is that time (previous
                # record to this one), which is what sps/learner_sps divide by.
                record['wall_seconds'] = elapsed-previous_elapsed
                record['overlap'] = overlap
                record['pool'] = after_pool
                if a.scripted_bots:
                    # Scripted banks sit after the pool's policy banks; their
                    # hist_*_bank fields are n-averaged like every env metric.
                    n = float(flat.get('env/n', 0.0))
                    scripted = {}
                    for i, name in enumerate(a.scripted_bots):
                        bank = scripted_first_bank + i
                        games = round(float(flat.get(f'env/bot_n_bank_{bank}', 0.0)) * n)
                        wins = round(float(flat.get(f'env/bot_score_bank_{bank}', 0.0)) * n)
                        scripted[f'{name}-{a.scripted_level}'] = dict(
                            bank=bank, games=games, winrate=wins/games if games else None)
                    record['scripted'] = scripted
                    record['scripted_share'] = share
                record['free_fraction'] = free
                record['entropy_head_scale_applied'] = applied_scale
                # The native trainer's own per-update timers (seconds, averaged
                # over buffers): eval_gpu = inference graph + action copy-back,
                # eval_env = OpenMP game stepping, rollout = whole barrier.
                record['native_perf'] = {key: native['perf'].get(key) for key in ('rollout', 'eval_gpu', 'eval_env')}
                metrics.write(json.dumps(record, allow_nan=False)+'\n'); metrics.flush()
                for event in native_selfplay.events(before_pool, after_pool, ledger, counts['learner_steps']):
                    events.write(json.dumps(event, allow_nan=False)+'\n')
                events.flush()
                console = dict(step=record['step'], learner_steps=record['learner_steps'],
                               learner_sps=record['learner_sps'], matches=record['matches'],
                               elo=round(after_pool['elo'], 1), pool=after_pool['size'])
                if a.scripted_bots:
                    console['scripted'] = {name: (None if row['winrate'] is None else round(row['winrate'], 3))
                                           for name, row in record['scripted'].items()}
                print(json.dumps(console), flush=True)
                previous, previous_elapsed = counts, elapsed
                if counts['learner_steps'] >= next_checkpoint:
                    save()
                    next_checkpoint = (counts['learner_steps']//a.checkpoint_every+1)*a.checkpoint_every
                status('running')
                if not a.no_eval and counts['learner_steps'] >= next_eval and not stop:
                    evaluate(save())
                    next_eval = (counts['learner_steps']//a.eval_every+1)*a.eval_every
            final = save()
            if not a.no_eval and not stop and counts['game_steps'] not in evaluated_steps:
                evaluate(final)
            finish_evaluation()
        status('stopped' if stop else 'complete', checkpoint=str(final))
    except BaseException as error:
        for process, log in pending:
            process.terminate()
            log.close()
        status('failed', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        if trainer is not None:
            _C.close(trainer)


if __name__ == '__main__':
    main()
