"""Matched short strength comparison of two native encoders (SET vs linear).

Protocol (pre-registered; every arm gets identical settings except --encoder):

1. Training. For each encoder and each seed, one fresh `puffer/train.py` run:
   16 games, 4 banks, horizon 128, learner passes 1, sparse reward (+1/-1,
   timeout -0.5, no kill shaping), episode cap 7200, maps 0,1,2 sampled at
   reset, entropy coefficient .01 decaying exponentially in learner decisions
   with a 30M half-life to the .0005 floor (train.py's default schedule, passed
   explicitly so a short run is a prefix of the long run's entropy schedule),
   learning rate .0003 held constant (`--anneal-lr 0`, train.py's default and
   the 400M run's setting; `--anneal-lr 1 --min-lr-ratio` selects the native
   cosine decay over each run's own budget instead, recorded in the plan),
   cudagraphs 1, the same game binary, `--steps` learner decisions,
   `--no-eval` (evaluation is matched below instead). Runs execute one at a time (one heavy job at a time).
2. Final evaluation per run (`evaluate.py --suite final`, CPU): the last
   checkpoint against `random` and the run's own initial checkpoint on maps
   0,1,2 plus `random` on held-out maps 3,4, side-swapped seed pairs, the
   same seeds for every run.
3. Head-to-head pool (CPU): every final checkpoint of both arms plus the
   random anchor play a side-swapped fixed-seed round robin on maps 0,1,2 at
   cap 7200. Timeouts are draws. Bradley-Terry ratings anchored at random =
   0 Elo (same fit as research/elo_ladder.py).

Primary metric: the pooled SET-vs-linear score (win = 1, draw = 1/2) over all
cross-arm pairs, maps and seed pairs, with a descriptive Wilson interval.
Verdict rule (fixed before any run): SET is `qualified` when the interval's
upper bound is >= 0.5 (SET is not significantly worse than linear) AND SET's
pooled score against random is >= linear's minus 0.05; `not-qualified`
otherwise. The report also gives per-seed scores, Elo, win/timeout rates from
training telemetry, and the training wall-hours of each arm so strength per
wall-hour can be read off directly.

Usage (from the repository root, no other heavy job running):

    native/artifacts/rl/venv/bin/python native/rl/autoresearch/strength_check.py \
        --output native/artifacts/rl/autoresearch/strength-set-vs-linear \
        --game <player> --steps 10000000 --seeds 73,173,273

Resuming is fail-closed: an existing output directory must carry an identical
plan (same settings, game payload hashes and source revision; only the creation
time may differ). A training run is reused only when it completed its budget
with the planned settings and game hashes and its final checkpoint still matches
its sidecar hash; a final evaluation is reused only for that same checkpoint
identity; the pool cache is keyed on every participant's checkpoint identity.
Anything else raises instead of mixing evidence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESEARCH = ROOT/'native/rl/research'
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT/'native/tools'))
from common import atomic_json  # noqa: E402
from evaluation_protocol import checkpoint_identity, map_ids, schedule  # noqa: E402
from reference import resolve_game, player_sources  # noqa: E402

TRAINER = ROOT/'native/rl/puffer/train.py'
EVALUATOR = RESEARCH/'evaluate.py'
OPENSSL = Path('/home/mte/Documents/Coding/bopl-re/openssl-unity-sha1.cnf')
ARMS = ('linear', 'set')


def wilson(successes, total, z=1.96):
    if total == 0: return (0., 1.)
    p = successes/total; d = 1+z*z/total
    c = (p+z*z/(2*total))/d; h = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return (max(0., c-h), min(1., c+h))


def environment():
    env = dict(os.environ)
    if OPENSSL.exists(): env.setdefault('OPENSSL_CONF', str(OPENSSL))
    return env


def complete(run, plan, arm, seed):
    """The run's result when it finished its budget under the planned settings; None when absent."""
    result = run/'result.json'
    if not result.exists(): return None
    data = json.loads(result.read_text())
    if data.get('status') != 'complete' or data.get('learner_steps', 0) < plan['steps']:
        raise RuntimeError(f'{run} exists but did not complete its budget; remove it to retrain')
    config = json.loads((run/'config.json').read_text())
    expected = dict(encoder=arm, seed=seed, workers=plan['workers'], banks=plan['banks'],
                    learner_passes=plan['learner_passes'], episode_ticks=plan['episode_ticks'],
                    training_maps=plan['training_maps'], learner_steps=plan['steps'],
                    entropy=plan['entropy'], entropy_floor=plan['entropy_floor'],
                    entropy_schedule='exponential', entropy_half_life=plan['entropy_half_life'],
                    learning_rate_schedule='cosine' if plan['anneal_lr'] else 'constant')
    mismatch = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    native_train = (config.get('native') or {}).get('train') or {}
    for k, v in dict(horizon=plan['horizon'], learning_rate=plan['learning_rate'],
                     anneal_lr=plan['anneal_lr'], min_lr_ratio=plan['min_lr_ratio']).items():
        if native_train.get(k) != v: mismatch[k] = (native_train.get(k), v)
    runtime = config.get('runtime') or config.get('game_runtime') or {}
    if plan['game_identity'] != {k: runtime.get(k) for k in plan['game_identity']}:
        mismatch['game_identity'] = (runtime, plan['game_identity'])
    if mismatch: raise RuntimeError(f'{run} was trained with different settings: {mismatch}')
    checkpoint = Path(data['checkpoint'])
    if checkpoint.parent.parent.resolve() != run.resolve():
        raise RuntimeError(f'{run} result points outside the run: {checkpoint}')
    data['checkpoint_identity'] = checkpoint_identity(checkpoint)  # raises on a replaced or corrupt .bin
    return data


def train(args, plan, arm, seed, game):
    run = args.output/f'{arm}-s{seed}'
    done = complete(run, plan, arm, seed)
    if done: return run, done
    if run.exists(): raise RuntimeError(f'{run} exists without a result; remove it to retrain')
    command = [sys.executable, str(TRAINER), '--output', str(run), '--game', str(game),
               '--steps', str(args.steps), '--workers', str(args.workers), '--banks', str(args.banks),
               '--horizon', str(args.horizon), '--seed', str(seed), '--encoder', arm,
               '--learner-passes', str(args.learner_passes),
               '--learning-rate', repr(args.learning_rate), '--anneal-lr', str(args.anneal_lr),
               '--min-lr-ratio', repr(args.min_lr_ratio), '--entropy', str(args.entropy),
               '--entropy-floor', str(args.entropy_floor), '--entropy-half-life', str(args.entropy_half_life),
               '--episode-ticks', str(args.episode_ticks),
               '--training-maps', ','.join(map(str, args.training_maps)),
               '--checkpoint-every', str(args.checkpoint_every), '--cudagraphs', str(args.cudagraphs), '--no-eval']
    log = args.output/f'{arm}-s{seed}-train.log'
    with log.open('w') as stream:
        stream.write(' '.join(command)+'\n'); stream.flush()
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=environment()).returncode
    if code: raise RuntimeError(f'training {arm} seed {seed} exited {code}; see {log}')
    done = complete(run, plan, arm, seed)
    if not done: raise RuntimeError(f'{run} did not complete its budget')
    return run, done


def final_evaluation(args, run, result, game):
    step = result['step']
    out = run/'final-evaluations'/f'{step:09d}'/'result.json'
    initial = run/'checkpoints/policy-000000000.bin'
    if out.exists():
        previous = json.loads(out.read_text())
        expected = len(schedule(args.training_maps, args.heldout_maps, [], args.final_games, args.final_heldout_games,
                                2, 50001, 60001, 70001)[0])
        if (previous.get('schema') != 2 or previous.get('suite') != 'final'
                or previous.get('checkpoint_identity') != result['checkpoint_identity']
                or previous.get('baseline_identity') != checkpoint_identity(initial)
                or previous.get('evaluation_maps') != args.training_maps or previous.get('heldout_maps') != args.heldout_maps
                or (previous.get('seed_base'), previous.get('heldout_seed_base'), previous.get('selfplay_seed')) != (50001, 60001, 70001)
                or len(previous.get('games', [])) != expected):
            raise RuntimeError(f'{out} is a final evaluation of a different checkpoint or schedule; remove it to re-evaluate')
        return previous
    command = [sys.executable, str(EVALUATOR), '--run', str(run), '--checkpoint', result['checkpoint'],
               '--initial', str(initial), '--suite', 'final', '--game', str(game),
               '--evaluation-maps', ','.join(map(str, args.training_maps)),
               '--heldout-maps', ','.join(map(str, args.heldout_maps)),
               '--games-per-opponent', str(args.final_games), '--heldout-games', str(args.final_heldout_games),
               '--frozen-games', '2', '--seed-base', '50001', '--heldout-seed-base', '60001',
               '--selfplay-seed', '70001', '--workers', str(args.evaluation_workers)]
    log = run.parent/f'{run.name}-final-evaluation.log'
    with log.open('w') as stream:
        code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=environment()).returncode
    if code: raise RuntimeError(f'final evaluation of {run.name} exited {code}; see {log}')
    return json.loads(out.read_text())


def telemetry(run):
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    last = records[-1]; totals = last['behavior']['totals']
    tail = records[-max(1, len(records)//10):]  # last 10% of updates
    episodes = sum(r['behavior']['totals']['episodes'] for r in tail) or 1
    return dict(updates=len(records), learner_steps=last['learner_steps'], physical_steps=last['step'],
                training_seconds=last['elapsed'], entropy_last=last['entropy'],
                explained_variance_last=last['explained_variance'],
                tail_win_rate=sum(r['behavior']['totals']['wins'] for r in tail)/episodes,
                tail_timeout_rate=sum(r['behavior']['totals']['timeouts'] for r in tail)/episodes,
                tail_mean_reward=float(np.mean([r['mean_reward'] for r in tail])),
                tail_kills_per_episode=sum(r['behavior']['totals'].get('kills', 0) for r in tail)/episodes,
                episodes_last_record=totals['episodes'])


def play_pool(args, participants, game):
    """Side-swapped fixed-seed round robin; returns per-game records."""
    from elo_ladder import Model, Random
    from environment import Environment, configure
    out = args.output/'pool'; out.mkdir(exist_ok=True)
    cache = out/'games.json'
    plan = dict(participants={n: dict(path=p, **checkpoint_identity(Path(p))) for n, p in participants.items()},
                maps=args.training_maps, games_per_pair=args.pool_games, seed_base=args.pool_seed_base,
                episode_ticks=args.episode_ticks)
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['plan'] != plan:
            raise RuntimeError(f'{cache} was played with different participants or settings; remove it to replay')
        return saved['games']
    configure(out/'workers', game)
    models = {'random': Random()}
    for name, path in participants.items():
        models[name] = Model(name, path, 'checkpoint')
    names = sorted(models)
    tasks = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            for map_id in args.training_maps:
                for pair in range(args.pool_games//2):
                    seed = args.pool_seed_base+pair
                    tasks.append(dict(a=a, b=b, map=map_id, seed=seed, side=0))
                    tasks.append(dict(a=a, b=b, map=map_id, seed=seed, side=1))

    def play(bucket):
        results = []
        with Environment(seed=args.pool_seed_base+len(tasks), limit=args.episode_ticks) as env:
            for task in bucket:
                env.reset(task['seed'], task['map'])
                rng = np.random.default_rng((task['seed'], task['map'], task['side'],
                                             int(hashlib.sha256((task['a']+task['b']).encode()).hexdigest()[:8], 16)))
                actors = {task['side']: models[task['a']].start(task, task['side'], rng),
                          1-task['side']: models[task['b']].start(task, 1-task['side'], rng)}
                while not env.ended:
                    observation = env.obs.copy()
                    env.step(np.stack([actors[0](observation), actors[1](observation)]))
                final = env.frame()
                timeout = bool(final['flags'] & 2)
                a_won = bool(final['players'][task['side']]['flags'] & 2) and not timeout
                b_won = bool(final['players'][1-task['side']]['flags'] & 2) and not timeout
                results.append(dict(task, ticks=final['t'],
                                    outcome='draw' if timeout or a_won == b_won else ('a' if a_won else 'b')))
        return results

    buckets = [tasks[i::args.pool_workers] for i in range(min(args.pool_workers, len(tasks)))]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.pool_workers) as executor:
        games = [g for bucket in executor.map(play, buckets) for g in bucket]
    atomic_json(cache, dict(plan=plan, elapsed=time.monotonic()-started, games=games))
    return games


def score(games, a_names, b_names):
    """Pooled score of the `a` side (win 1, draw 1/2) over games between the two name sets."""
    wins = draws = total = 0
    for g in games:
        if g['a'] in a_names and g['b'] in b_names: mine, theirs = 'a', 'b'
        elif g['a'] in b_names and g['b'] in a_names: mine, theirs = 'b', 'a'
        else: continue
        total += 1
        if g['outcome'] == mine: wins += 1
        elif g['outcome'] == 'draw': draws += 1
    s = wins+draws/2
    low, high = wilson(s, total) if total else (0., 1.)
    return dict(games=total, wins=wins, draws=draws, losses=total-wins-draws,
                score=s/total if total else None, interval=[low, high])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game', type=Path)
    parser.add_argument('--steps', type=int, default=10_000_000, help='Learner decisions per run')
    parser.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=[73, 173, 273])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--banks', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--learner-passes', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=.0003, help='train.py default')
    parser.add_argument('--anneal-lr', type=int, choices=(0, 1), default=0,
                        help='0 keeps the learning rate constant (train.py default); 1 uses the native cosine decay over each run budget')
    parser.add_argument('--min-lr-ratio', type=float, default=0., help='Final/base learning-rate ratio when --anneal-lr 1')
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--entropy-floor', type=float, default=.0005)
    parser.add_argument('--entropy-half-life', type=int, default=30_000_000,
                        help='Learner decisions per halving of the entropy coefficient (train.py default)')
    parser.add_argument('--episode-ticks', type=int, default=7200)
    parser.add_argument('--training-maps', type=map_ids, default=[0, 1, 2])
    parser.add_argument('--heldout-maps', type=map_ids, default=[3, 4])
    parser.add_argument('--checkpoint-every', type=int, default=2_500_000)
    parser.add_argument('--cudagraphs', type=int, default=1)
    parser.add_argument('--final-games', type=int, default=16, help='Per opponent per map in the final evaluation')
    parser.add_argument('--final-heldout-games', type=int, default=8)
    parser.add_argument('--evaluation-workers', type=int, default=8)
    parser.add_argument('--pool-games', type=int, default=16, help='Per pair per map (even; half per side)')
    parser.add_argument('--pool-seed-base', type=int, default=80001)
    parser.add_argument('--pool-workers', type=int, default=16)
    args = parser.parse_args()
    if args.pool_games % 2 or args.pool_games < 2 or args.final_games % 2 or args.final_heldout_games % 2:
        parser.error('game counts must be even side-swapped seed pairs')
    if set(args.training_maps) & set(args.heldout_maps): parser.error('held-out maps overlap training maps')
    if len(set(args.seeds)) != len(args.seeds): parser.error('distinct seeds required')
    game = resolve_game(args.game)
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(schema=1, kind='bopl-native-strength-check', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                game=str(game), game_identity={k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in player_sources(game).items()},
                arms=list(ARMS), **{k: v for k, v in vars(args).items() if k not in ('output', 'game')},
                verdict_rule='qualified iff upper Wilson bound of pooled SET-vs-linear score >= 0.5 and SET score vs random >= linear score vs random - 0.05')
    existing = args.output/'plan.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        differences = {k: (previous.get(k), v) for k, v in plan.items() if k != 'created_utc' and previous.get(k) != v}
        if differences or set(previous) != set(plan):
            raise RuntimeError(f'{existing} differs from this invocation (use a fresh --output or the same settings): {differences}')
    else:
        atomic_json(existing, plan)

    runs = {}
    for seed in args.seeds:            # interleave arms so drift in the machine affects both equally
        for arm in ARMS:
            print(f'== train {arm} seed {seed} {time.strftime("%H:%M:%S")}', flush=True)
            run, result = train(args, plan, arm, seed, game)
            runs[(arm, seed)] = dict(run=run, result=result)
    for (arm, seed), item in runs.items():
        print(f'== final evaluation {arm} seed {seed} {time.strftime("%H:%M:%S")}', flush=True)
        item['final'] = final_evaluation(args, item['run'], item['result'], game)
        item['telemetry'] = telemetry(item['run'])
    participants = {f'{arm}-s{seed}': item['result']['checkpoint'] for (arm, seed), item in runs.items()}
    print(f'== pool {len(participants)+1} participants {time.strftime("%H:%M:%S")}', flush=True)
    games = play_pool(args, participants, game)

    from elo_ladder import bradley_terry
    records = {}
    for g in games:
        row = records.setdefault((g['a'], g['b']), [0, 0, 0]); row[{'a': 0, 'b': 1, 'draw': 2}[g['outcome']]] += 1
    ratings = bradley_terry(sorted({'random', *participants}), records, 'random')
    by_arm = {arm: {n for n in participants if n.startswith(arm+'-')} for arm in ARMS}
    head_to_head = score(games, by_arm['set'], by_arm['linear'])
    per_seed = {str(seed): score(games, {f'set-s{seed}'}, {f'linear-s{seed}'}) for seed in args.seeds}
    vs_random = {arm: score(games, by_arm[arm], {'random'}) for arm in ARMS}
    qualified = (head_to_head['interval'][1] >= .5
                 and vs_random['set']['score'] >= vs_random['linear']['score']-.05)
    arms = {}
    for arm in ARMS:
        items = [runs[(arm, s)] for s in args.seeds]
        arms[arm] = dict(
            runs=[dict(seed=s, checkpoint=runs[(arm, s)]['result']['checkpoint'],
                       checkpoint_identity=checkpoint_identity(Path(runs[(arm, s)]['result']['checkpoint'])),
                       training_wall_hours=runs[(arm, s)]['result']['elapsed']/3600, matches=runs[(arm, s)]['result']['matches'],
                       elo=round(ratings[f'{arm}-s{s}'], 1), telemetry=runs[(arm, s)]['telemetry'],
                       final_groups=runs[(arm, s)]['final']['groups']) for s in args.seeds],
            mean_training_wall_hours=float(np.mean([i['result']['elapsed'] for i in items]))/3600,
            mean_elo=float(np.mean([ratings[f'{arm}-s{s}'] for s in args.seeds])),
            final_win_rate_vs_random=float(np.mean([g['win_rate'] for i in items for g in i['final']['groups'] if g['opponent'] == 'random' and g['map'] in args.training_maps])),
            final_win_rate_vs_random_heldout=float(np.mean([g['win_rate'] for i in items for g in i['final']['groups'] if g['opponent'] == 'random' and g['map'] in args.heldout_maps])),
            final_win_rate_vs_initial=float(np.mean([g['win_rate'] for i in items for g in i['final']['groups'] if g['opponent'] == 'initial'])),
            pool_score_vs_random=vs_random[arm])
    report = dict(schema=1, plan=plan, verdict='qualified' if qualified else 'not-qualified',
                  set_vs_linear=head_to_head, set_vs_linear_per_seed=per_seed, ratings={k: round(v, 1) for k, v in ratings.items()},
                  pool_games=len(games), pool_timeouts=sum(g['outcome'] == 'draw' for g in games), arms=arms)
    atomic_json(args.output/'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'plan'}, indent=1))
    print(f"VERDICT {report['verdict']}: SET vs linear score {head_to_head['score']:.3f} "
          f"[{head_to_head['interval'][0]:.3f}, {head_to_head['interval'][1]:.3f}] over {head_to_head['games']} games; "
          f"Elo set {arms['set']['mean_elo']:.0f} vs linear {arms['linear']['mean_elo']:.0f}; "
          f"wall-hours per run set {arms['set']['mean_training_wall_hours']:.2f} vs linear {arms['linear']['mean_training_wall_hours']:.2f}")


if __name__ == '__main__':
    main()
