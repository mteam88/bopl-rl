"""Aim-head experiment: autoregressive buttons->aim and Fourier aim logits.

Four arms x three seeds at the overnight-6 fixed settings, 200M learner
decisions each (research-aim-20260912/AIM_LANDSCAPE.md, decided 2026-09-12):

  baseline  --aim-fourier 0  --aim-ar 0
  ar        --aim-fourier 0  --aim-ar 1   buttons sampled first, aim conditioned on the combo
  fourier   --aim-fourier 16 --aim-ar 0   256 aim logits from 16 harmonics + a neutral logit
  both      --aim-fourier 16 --aim-ar 1

`launch` runs the twelve trainings one after another as systemd scopes (the
sweep driver's recipe: Rust sim, PufferLib from this tree, passive OpenMP),
optionally after another user unit has finished, and skips runs that already
completed so it can be restarted. `report` reads what the runs wrote and
prints the pre-registered judgement per arm:

  * aim-head entropy at 50M and 100M learner decisions (metrics.jsonl);
  * arrow hit rate over the evaluation-map groups of each evaluation
    (train/sweep/flags.evaluation_metrics), pooled over the evaluations at or
    after 100M;
  * median 15 Hz aim change in degrees from the saved evaluation replays
    (eval/run_insights), same pooling;
  * league rating (league candidate, <run>/league/<checkpoint>.json), last
    evaluation and IQM-free mean of the last three, as the tiebreaker.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'train/sweep')]
import packed  # noqa: E402
PUFFERLIB_DIR = sorted((ROOT / 'native/artifacts/rl/dependencies').glob('PufferLib-*'))[-1]
VENV_PYTHON = ROOT / 'native/artifacts/rl/venv/bin/python'
TRAIN = ROOT / 'train/train.py'
BASE_CONFIG = ROOT / 'train/sweep/configs/overnight-6.toml'
DEFAULT_ROOT = ROOT / 'native/artifacts/rl/sweeps/aim-ar-fourier'

ARMS = {'baseline': (0, 0), 'ar': (0, 1), 'fourier': (16, 0), 'both': (16, 1),
        # K sweep (2026-09-13, after Fourier 16 won the four-arm comparison): Fourier only.
        'k1': (1, 0), 'k2': (2, 0), 'k3': (3, 0), 'k4': (4, 0), 'k8': (8, 0), 'k32': (32, 0), 'k64': (64, 0)}
MAIN_ARMS = ('baseline', 'ar', 'fourier', 'both')
SEEDS = (73, 74, 75)
STEPS = 200_000_000
CHECKPOINT_EVERY = 25_000_000
EVAL_EVERY = 25_000_000
ENTROPY_MARKS = (50_000_000, 100_000_000, 200_000_000)
POOL_FROM = 100_000_000          # evaluations at or after this many learner decisions are pooled


def log(message):
    print(f'[aim {time.strftime("%H:%M:%S")}] {message}', flush=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return default


def run_name(arm, seed):
    return f'{arm}-s{seed}'


def runs(arms, seeds):
    return [(arm, seed) for arm in arms for seed in seeds]


# ---- launch ---------------------------------------------------------------

def arguments(spec, arm, seed, output, steps):
    fourier, autoregressive = ARMS[arm]
    spec.fixed.update(steps=steps, checkpoint_every=CHECKPOINT_EVERY, eval_every=EVAL_EVERY,
                      aim_fourier=fourier, aim_ar=autoregressive)
    return spec.trial_arguments({}, output, seed=seed)


def command(spec, arm, seed, output, steps, memory_max, nice):
    unit = f'bopl-aim-{run_name(arm, seed)}'
    return ['systemd-run', '--user', '--scope', '--collect', '-p', f'MemoryMax={memory_max}',
            '--unit', unit, '--', 'nice', '-n', str(nice), str(VENV_PYTHON), str(TRAIN),
            *arguments(spec, arm, seed, output, steps)]


def environment():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(PUFFERLIB_DIR)
    env.setdefault('BOPL_RL_BACKEND', 'rust')
    env.setdefault('BOPL_OMP_THREADS', str(os.cpu_count() or 12))
    env.setdefault('OMP_WAIT_POLICY', 'passive')
    env.setdefault('BOPL_ENV_THREADS', str(max(2, (os.cpu_count() or 12) - 4)))
    return env


def unit_active(unit):
    result = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True)
    return result.stdout.strip() in ('active', 'activating', 'deactivating')


def completed(output):
    return read_json(output / 'status.json', {}).get('status') == 'complete'


def launch(args):
    import sweep_config
    spec = sweep_config.load(args.base)
    args.root.mkdir(parents=True, exist_ok=True)
    state_path = args.root / 'state.json'
    state = read_json(state_path, {'runs': {}})
    if args.wait_for:
        while unit_active(args.wait_for):
            log(f'waiting for {args.wait_for} to finish')
            time.sleep(args.poll_seconds)
    for arm, seed in runs(args.arms, args.seeds):
        name = run_name(arm, seed)
        output = args.root / name
        if completed(output):
            log(f'{name}: already complete')
            continue
        if output.exists() and not args.dry_run:
            raise SystemExit(f'{output} exists but is not complete; move it aside or finish it manually')
        argv = command(spec, arm, seed, output, args.steps, args.memory_max, args.nice)
        if args.dry_run:
            print(' '.join(argv))
            continue
        log(f'launching {name}')
        started = time.time()
        with (args.root / f'{name}.log').open('a') as log_file:
            process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT,
                                       env=environment(), cwd=ROOT)
            state['runs'][name] = dict(arm=arm, seed=seed, status='running', pid=process.pid,
                                       launched_at=started, run_dir=str(output))
            state_path.write_text(json.dumps(state, indent=1) + '\n')
            code = process.wait()
        state['runs'][name].update(status='complete' if code == 0 and completed(output) else 'failed',
                                   exit_code=code, finished_at=time.time(),
                                   elapsed=time.time() - started)
        state_path.write_text(json.dumps(state, indent=1) + '\n')
        log(f'{name}: exit {code} after {(time.time() - started) / 60:.1f} min')
        if code != 0 and not args.keep_going:
            raise SystemExit(f'{name} failed (exit {code}); see {args.root / (name + ".log")}')
    log('all runs launched')


# ---- report ---------------------------------------------------------------

def metrics_at(run_dir, key, marks):
    """Value of `key` at the first metrics row at or after each learner-decision mark."""
    found = {mark: None for mark in marks}
    try:
        with packed.open_text(Path(run_dir) / 'metrics.jsonl') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                steps = row.get('learner_steps')
                if steps is None:
                    continue
                for mark in marks:
                    if found[mark] is None and steps >= mark:
                        found[mark] = (steps, row.get(key))
    except FileNotFoundError:
        pass
    return found


def learner_sps(run_dir):
    values = []
    try:
        with packed.open_text(Path(run_dir) / 'metrics.jsonl') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    break
                if row.get('learner_sps'):
                    values.append(row['learner_sps'])
    except FileNotFoundError:
        pass
    return statistics.median(values) if values else None


def evaluations(run_dir):
    """[(learner_steps, directory, report)] for every complete evaluation of the run."""
    import scoring
    rows = []
    for result_file in sorted(Path(run_dir).glob('evaluations/*/result.json')):
        report = read_json(result_file)
        if not report or not report.get('checkpoint'):
            continue
        steps = scoring._checkpoint_learner_steps(run_dir, report['checkpoint'])
        if steps is None:
            continue
        rows.append((steps, result_file.parent, report))
    return sorted(rows, key=lambda row: row[0])


def rating_of(run_dir, report, reference):
    import candidate
    checkpoint = Path(run_dir) / 'checkpoints' / report['checkpoint']
    payload = candidate.load(candidate.result_path(checkpoint), reference)
    return payload['rating'] if payload else None


def summarize_run(run_dir, reference):
    import flags
    import run_insights
    entropy = metrics_at(run_dir, 'entropy_aim', ENTROPY_MARKS)
    fired = hits = 0.0
    aim_changes = []
    ratings = []
    pooled = 0
    for steps, directory, report in evaluations(run_dir):
        rating = rating_of(run_dir, report, reference)
        ratings.append((steps, rating))
        if steps < POOL_FROM:
            continue
        pooled += 1
        metrics = flags.evaluation_metrics(report)
        fired += metrics['arrows_per_game'] * metrics['games']
        hits += metrics['arrow_hits_per_game'] * metrics['games']
        insights = read_json(directory / 'insights.json')
        if not insights or insights.get('schema') != run_insights.SCHEMA:
            insights = run_insights.evaluate_directory(directory)
        for game in report['games']:
            import visuals
            if not game.get('visual') or not visuals.exists(directory / game['visual']):
                continue
            visual = visuals.load(directory / game['visual'])
            _, changes, _, _ = run_insights.game_insights(visual, game['side'])
            aim_changes += changes
    rated = [r for _, r in ratings if r is not None]
    return dict(
        status=read_json(Path(run_dir) / 'status.json', {}).get('status'),
        learner_steps=read_json(Path(run_dir) / 'status.json', {}).get('learner_steps'),
        entropy_aim={str(mark): (value[1] if value else None) for mark, value in entropy.items()},
        evaluations=len(ratings), pooled_evaluations=pooled,
        arrows=fired, arrow_hits=hits,
        arrow_hit_rate=(hits / fired) if fired else None,
        aim_change_degrees=run_insights.median(aim_changes), aim_change_decisions=len(aim_changes),
        elo_last=rated[-1] if rated else None,
        elo_last3=(sum(rated[-3:]) / len(rated[-3:])) if rated else None,
        elo_curve=ratings, learner_sps=learner_sps(run_dir))


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0, len(values))


def fmt(value, digits=3):
    return '-' if value is None else f'{value:.{digits}f}'


def cell(values, digits=3):
    mean, sd, n = mean_sd(values)
    if not n:
        return '-'
    return f'{mean:.{digits}f} ± {sd:.{digits}f} (n={n})'


def report(args):
    sys.path.insert(0, str(ROOT / 'league'))
    import candidate
    reference = candidate.reference()
    per_run = {}
    for arm, seed in runs(args.arms, args.seeds):
        name = run_name(arm, seed)
        run_dir = args.root / name
        if not run_dir.exists():
            continue
        per_run[name] = dict(arm=arm, seed=seed, **summarize_run(run_dir, reference))
    (args.root / 'report.json').write_text(json.dumps(per_run, indent=1) + '\n')
    lines = ['| arm | K | AR | aim H @50M | aim H @100M | aim H @200M | arrow hit rate ≥100M | median aim Δ° ≥100M | Elo last | Elo last-3 | learner/s |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in args.arms:
        rows = [r for r in per_run.values() if r['arm'] == arm]
        if not rows:
            continue
        K, AR = ARMS[arm]
        lines.append('| ' + ' | '.join([
            arm, str(K), str(AR),
            cell([r['entropy_aim']['50000000'] for r in rows]),
            cell([r['entropy_aim']['100000000'] for r in rows]),
            cell([r['entropy_aim']['200000000'] for r in rows]),
            cell([r['arrow_hit_rate'] for r in rows]),
            cell([r['aim_change_degrees'] for r in rows], 1),
            cell([r['elo_last'] for r in rows], 0),
            cell([r['elo_last3'] for r in rows], 0),
            cell([r['learner_sps'] for r in rows], 0)]) + ' |')
    lines += ['', '| run | status | steps | aim H 50M/100M/200M | hit rate | aim Δ° (n) | Elo curve |', '|---|---|---|---|---|---|---|']
    for name, r in per_run.items():
        curve = ' '.join(f'{s // 1_000_000}M:{fmt(v, 0)}' for s, v in r['elo_curve'])
        lines.append(f"| {name} | {r['status']} | {r['learner_steps']} | "
                     f"{fmt(r['entropy_aim']['50000000'])}/{fmt(r['entropy_aim']['100000000'])}/{fmt(r['entropy_aim']['200000000'])} | "
                     f"{fmt(r['arrow_hit_rate'])} ({r['arrows']:.0f} arrows) | {fmt(r['aim_change_degrees'], 1)} ({r['aim_change_decisions']}) | {curve} |")
    text = '\n'.join(lines) + '\n'
    (args.root / 'report.md').write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--arms', type=lambda s: s.split(','), default=list(MAIN_ARMS),
                        help='Comma-separated arm names from ' + ','.join(ARMS))
    parser.add_argument('--seeds', type=lambda s: [int(x) for x in s.split(',')], default=list(SEEDS))
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('launch')
    p.add_argument('--base', type=Path, default=BASE_CONFIG, help='Sweep TOML whose [fixed] block is the shared setting')
    p.add_argument('--steps', type=int, default=STEPS)
    p.add_argument('--wait-for', help='User unit that must be inactive before the first run starts')
    p.add_argument('--poll-seconds', type=int, default=120)
    p.add_argument('--memory-max', default='6G')
    p.add_argument('--nice', type=int, default=10)
    p.add_argument('--keep-going', action='store_true', help='Continue with the next run after a failure')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=launch)
    p = sub.add_parser('report')
    p.set_defaults(func=report)
    args = parser.parse_args()
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        parser.error(f'unknown arms {sorted(unknown)}')
    args.func(args)


if __name__ == '__main__':
    main()
