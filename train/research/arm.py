"""Research arms (native/rl/research/PLAN-2026-10.md, REVIEW-2026-10-cursor.md): one experiment is one TOML file.

    venv/bin/python train/research/arm.py train/research/arms/e0.toml [more.toml ...] [--dry-run] [--only NAME,...]

An arm warm-starts from `init`, trains `constant_steps` learner steps at constant LR with
--ema-decay (the trainer keeps one EMA; its decay is computed per arm so the half-life is
`ema_half_life` learner steps whatever the batch per optimizer step: 5B gives 0.9999966 at the l2
batch; an explicit `ema_decay` overrides), then runs the matched cosine anneal (`anneal_steps`, --init-weights <last raw
checkpoint>, same seed) down to the absolute floor `anneal_floor_lr` (--min-lr-ratio is computed
from the arm's --learning-rate), then scores its endpoints with league/all_combos.py --games-out:
raw, EMA and annealed, plus the raw checkpoints at `raw_at` learner steps.

Training flags are the l2 main's turn-100 command (`base`, a command.txt) minus the flags an arm
owns (OWNED) and minus the `exclude_opponents` members; `flags` adds or overrides (true = bare flag,
false = removed). Top-level keys are defaults, [arm.<name>] tables override them (`flags` merges);
`seeds = [...]` expands an arm into replicate arms <name>-s<seed> of group <name>; `compare = "<group>"` names the control group the
report tests it against on their common seeds; top-level `order = "interleave"` trains one arm of
each group in turn (control s301, treatment s301, control s302, ...). A [factorial]
table adds one arm per run of a two-level (fractional) factorial: [factorial.factors.<letter>] with
`low`/`high` flag tables, `generators = { E = "ABCD" }` for the aliased factors, `seed` (run i gets
seed + i), `replicates`, and `order_seed` (runs are trained in a shuffled order).

Evaluation: every endpoint plays the held-out `panel` (league members, none of them in the training
opponents: the run refuses otherwise), `games` per endpoint split evenly over it; the primary metric
is the equal-weight panel mean. [opponents.<name>] adds secondary opponents (`path`: a file or a
league member, `games`, `endpoints`). Seeds: one seed base per purpose, screen 211001 and
confirm/release 221001. Endpoint e against opponent o (panel first, then [opponents] in order)
plays seed base + SEED_STRIDE*(SLOTS*o + e), so an arm's endpoints play fresh setups and every arm
plays the same setups per endpoint; append new opponents and endpoints at the end.

Arms train one at a time (each stage a systemd unit, CPUWeight 1000, after waiting for `gpu_mib`
free on the card); evals run one at a time in CPUWeight 30 units, overlapping the next arm's
training. Rerunning skips every stage whose output on disk is complete: a training stage is done
when its status.json says complete, and resumes with --resume from its last checkpoint otherwise;
an eval is done when its --out row has all its games. Every command is appended to <out>/commands.log.
"""
import argparse
import functools
import itertools
import json
import math
import queue
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / 'native/artifacts/rl/venv/bin/python'
PUFFER = ROOT / 'native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9'
ENDPOINTS = ('raw', 'ema', 'anneal')
SEED_STRIDE, SLOTS = 50_000, 8
SEED_BASES = dict(screen=211001, confirm=221001, release=221001)
MAX_PAIRS = 10_000   # below the 10000 between the two bases, so screen and confirm setups never meet
LEARNER_ROWS_PER_WORKER = 1.5   # learner rows per worker per tick at the l2 bank layout (config.json minibatch_size)
PANEL = ['final5-18666m', 'anneal20-26666m', 'team5b-6250m', 'g111-20b-26027m']
OWNED = ('--output', '--steps', '--init-weights', '--anneal-lr', '--min-lr-ratio', '--seed', '--checkpoint-every',
         '--metrics-every', '--ema-decay', '--resume')
PROGRAM_KEYS = ('name', 'out', 'purpose', 'seed_base', 'eval_workers', 'factorial', 'order')
DEFAULTS = dict(
    purpose='screen', eval_workers=6, gpu_mib=9000,
    base='native/artifacts/rl/sweeps/l2-anneal-2/command.txt',
    init='native/artifacts/rl/league-loops/l2/main/checkpoints/policy-136002994176.bin',
    constant_steps=6_000_000_000, anneal_steps=1_000_000_000, anneal_floor_lr=1.2e-5,
    ema_half_life=5_000_000_000, ema_decay=None,
    checkpoint_every=None, raw_at=[], metrics_every=10_000_000, flags={}, exclude_opponents=[],
    panel=PANEL, games=4300, opponents={})
NTFY = 'http://127.0.0.1:2586/bopl'


def log(message):
    print(f'[research {time.strftime("%m-%d %H:%M:%S")}] {message}', flush=True)


@functools.cache
def members():
    sys.path.insert(0, str(ROOT / 'league'))
    import registry
    return {m['name']: m for m in registry.load(verify=False)['members']}


def path(spec):
    """A path (absolute or from the repo root) or a league member name."""
    p = Path(spec) if Path(spec).is_absolute() else ROOT / spec
    if p.exists() or '/' in str(spec):
        return p
    if spec not in members() or not members()[spec].get('path'):
        raise SystemExit(f'{spec}: neither a file nor a league member with weights')
    return ROOT / members()[spec]['path']


def parse_flags(argv):
    flags, i = {}, 0
    while i < len(argv):
        flag, i = argv[i], i + 1
        values = []
        while i < len(argv) and not argv[i].startswith('--'):
            values.append(argv[i])
            i += 1
        flags[flag] = values
    return flags


def base_flags(command_file):
    argv = shlex.split(path(command_file).read_text())
    flags = parse_flags(argv[next(i for i, x in enumerate(argv) if x.endswith('train.py')) + 1:])
    return {k: v for k, v in flags.items() if k not in OWNED}


def factorial_arms(spec):
    """One arm table per run: the full factorial over the base factors, generated factors as products."""
    factors = spec['factors']
    generators = spec.get('generators', {})
    letters = [f for f in factors if f not in generators]
    runs = []
    for levels in itertools.product((-1, 1), repeat=len(letters)):
        x = dict(zip(letters, levels))
        for f, word in generators.items():   # "-ABCD" is the other half-fraction
            x[f] = (-1 if word.startswith('-') else 1) * math.prod(x[c] for c in word.lstrip('-'))
        runs.append({f: x[f] for f in factors})
    arms = []
    for r in range(spec.get('replicates', 1)):
        for i, x in enumerate(runs):
            flags = {}
            for f, level in x.items():
                flags.update(factors[f].get('high' if level > 0 else 'low', {}))
            name = ''.join(f.lower() for f in factors if x[f] > 0) or '1'
            arms.append((f"ff-{name}" + (f'-r{r}' if spec.get('replicates', 1) > 1 else ''),
                         dict(seed=spec['seed'] + r * len(runs) + i, group=f'ff-{name}', factors=x, flags=flags)))
    random.Random(spec.get('order_seed', 0)).shuffle(arms)
    return arms


def load_arms(config_file):
    raw = tomllib.loads(Path(config_file).read_text())
    program = raw['name']
    top = {**DEFAULTS, **{k: v for k, v in raw.items() if k not in ('arm', 'factorial')}}
    top['seed_base'] = top.get('seed_base', SEED_BASES[top['purpose']])
    out = path(top.get('out', f'native/artifacts/rl/research/{program}'))
    tables = list(raw.get('arm', {}).items()) + (factorial_arms(raw['factorial']) if 'factorial' in raw else [])
    arms = []
    for name, table in tables:
        cfg = {**top, **table, 'flags': {**top['flags'], **table.get('flags', {})}}
        if bad := [k for k in PROGRAM_KEYS if k in table]:
            raise SystemExit(f'{config_file}: arm {name} sets program keys {bad}')
        for seed in cfg.pop('seeds', None) or [cfg['seed']]:
            arm = dict(cfg, name=f'{name}-s{seed}' if 'seeds' in table else name, group=table.get('group', name),
                       seed=seed, program=program)
            arm['dir'] = out / arm['name']
            arm['checkpoint_every'] = arm['checkpoint_every'] or arm['constant_steps']
            flags = base_flags(arm['base'])
            for key, value in arm['flags'].items():
                flag = '--' + key.removeprefix('--')
                if flag in OWNED:
                    raise SystemExit(f'{config_file}: {flag} is set by the arm keys, not flags')
                if value is False:
                    flags.pop(flag, None)
                else:
                    flags[flag] = [] if value is True else [str(v) for v in (value if isinstance(value, list) else [value])]
            opponents = [o for o in ','.join(flags.get('--opponents', [])).split(',') if o not in arm['exclude_opponents']]
            if '--opponents' in flags:
                flags['--opponents'] = [','.join(opponents)]
            if leaked := set(arm['panel']) & set(opponents):
                raise SystemExit(f'{config_file}: arm {arm["name"]} trains against panel members {sorted(leaked)}; '
                                 'add them to exclude_opponents')
            arm['train_flags'] = flags
            if arm['ema_decay'] is None and arm['ema_half_life']:
                value = lambda flag, default: float(flags.get(flag, [default])[0])
                per_step = LEARNER_ROWS_PER_WORKER * value('--workers', 16) * value('--horizon', 128) / (
                    value('--minibatch-div', 1) * value('--learner-passes', 1))
                arm['ema_decay'] = round(0.5 ** (per_step / arm['ema_half_life']), 10)
            arm['min_lr_ratio'] = arm['anneal_floor_lr'] / float(flags['--learning-rate'][0])
            if not 0 < arm['min_lr_ratio'] <= 1:
                raise SystemExit(f'{config_file}: anneal_floor_lr is above the arm\'s --learning-rate')
            if arm['eval_workers'] > 6:
                raise SystemExit(f'{config_file}: eval_workers is at most 6')
            arm['endpoint_names'] = [*ENDPOINTS, *(f'raw{s // 1_000_000}m' for s in arm['raw_at'])]
            if len(arm['endpoint_names']) > SLOTS:
                raise SystemExit(f'{config_file}: at most {SLOTS - len(ENDPOINTS)} raw_at endpoints')
            per_member = 2 * round(arm['games'] / (2 * len(arm['panel'])))
            arm['evals'] = [(m, dict(path=m, games=per_member)) for m in arm['panel']] + list(arm['opponents'].items())
            for opponent, spec in arm['evals']:
                games = spec.get('games', arm['games'])
                if games % 2 or not 0 < games // 2 <= MAX_PAIRS:
                    raise SystemExit(f'{config_file}: {opponent} games must be even and at most {2 * MAX_PAIRS}')
            arms.append(arm)
    if top.get('order') == 'interleave':   # one arm of each group in turn, so drift hits every group alike
        groups = {}
        for arm in arms:
            groups.setdefault(arm['group'], []).append(arm)
        arms = [g[i] for i in range(max(map(len, groups.values()))) for g in groups.values() if i < len(g)]
    return out, arms


# ---- stages -----------------------------------------------------------------

def unit_name(arm, stage):
    return re.sub(r'[^A-Za-z0-9_.-]', '-', f"research-{arm['program']}-{arm['name']}-{stage}")


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit + '.service']).returncode == 0


def checkpoints(run):
    """Raw (non-EMA) checkpoints of a run by learner step."""
    return {int(p.stem.split('-')[1]): p for p in (run / 'checkpoints').glob('policy-*.bin') if not p.stem.endswith('-ema')}


class Runner:
    def __init__(self, out, dry_run):
        self.out, self.dry_run, self.lock = out, dry_run, threading.Lock()

    def unit(self, unit, shell, log_path, weight, memory, env):
        if self.dry_run:
            print(f'# {unit}\n{shell}\n', flush=True)
            return
        if active(unit):   # a previous runner's unit is still going: wait for it rather than start another
            log(f'{unit} is already running; waiting for it')
            while active(unit):
                time.sleep(30)
            return
        argv = ['systemd-run', '--user', '--quiet', '--wait', '--collect', f'--unit={unit}', '-p', f'MemoryHigh={memory}',
                '-p', f'CPUWeight={weight}', '-p', f'WorkingDirectory={ROOT}', '--setenv=BOPL_RL_BACKEND=rust',
                *[f'--setenv={e}' for e in env], 'bash', '-c', f'({shell}) >> {shlex.quote(str(log_path))} 2>&1']
        with self.lock:
            self.out.mkdir(parents=True, exist_ok=True)
            with open(self.out / 'commands.log', 'a') as f:
                f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {shlex.join(argv)}\n')
        subprocess.run(['systemctl', '--user', 'reset-failed', unit + '.service'], capture_output=True)
        log(f'{unit} started')
        code = subprocess.run(argv).returncode
        log(f'{unit} exited {code}')

    # -- training

    @staticmethod
    def finished(run):
        try:
            status = json.loads((run / 'status.json').read_text())
        except (OSError, ValueError):
            return None
        return Path(status['checkpoint']) if status.get('status') == 'complete' and Path(status['checkpoint']).exists() \
            else None

    def train(self, arm, stage, init):
        run, unit = arm['dir'] / stage, unit_name(arm, stage)
        if not self.dry_run and active(unit):   # a previous runner's unit: let it finish, then judge its output
            log(f'{unit} is already running; waiting for it')
            while active(unit):
                time.sleep(30)
        if done := self.finished(run):
            return done
        owned = {'--output': [run], '--init-weights': [init], '--seed': [arm['seed']],
                 '--metrics-every': [arm['metrics_every']]}
        if stage == 'constant':
            owned |= {'--steps': [arm['constant_steps']], '--checkpoint-every': [arm['checkpoint_every']]}
            if arm['ema_decay']:
                owned['--ema-decay'] = [arm['ema_decay']]
        else:
            owned |= {'--steps': [arm['anneal_steps']], '--checkpoint-every': [arm['anneal_steps']],
                      '--anneal-lr': [1], '--min-lr-ratio': [f"{arm['min_lr_ratio']:.6g}"]}
        if (run / 'pool/state.json').exists():
            owned['--resume'] = []
        elif run.exists() and not self.dry_run:
            run.rename(run.with_name(f'{stage}.crashed-{time.strftime("%Y%m%d-%H%M%S")}'))
        argv = [PY, ROOT / 'train/train.py']
        for flag, values in [*owned.items(), *arm['train_flags'].items()]:
            argv += [flag, *values]
        if not self.dry_run:
            arm['dir'].mkdir(parents=True, exist_ok=True)
            self.wait_gpu(arm['gpu_mib'], unit)
        self.unit(unit, shlex.join(map(str, argv)), arm['dir'] / f'{stage}.log', 1000, '12G', [f'PYTHONPATH={PUFFER}'])
        if self.dry_run:
            return run / 'checkpoints' / f'policy-<{stage} last>.bin'
        if not (done := self.finished(run)):
            raise RuntimeError(f'{unit} did not complete (see {arm["dir"]}/{stage}.log)')
        return done

    @staticmethod
    def wait_gpu(mib, unit):
        def free():
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True).stdout
            return int(out.split()[0])
        if free() < mib:
            log(f'{unit} waits for {mib} MiB free on the GPU')
            while free() < mib:
                time.sleep(60)

    # -- evaluation

    def endpoints(self, arm, raw, annealed):
        endpoints = dict(raw=raw, anneal=annealed)
        if arm['ema_decay']:
            endpoints['ema'] = raw.with_name(f'{raw.stem}-ema.bin')
        saved = checkpoints(arm['dir'] / 'constant')
        for step, name in zip(arm['raw_at'], arm['endpoint_names'][len(ENDPOINTS):]):
            later = [s for s in saved if s >= step]
            endpoints[name] = saved[min(later)] if later else arm['dir'] / f'constant/checkpoints/policy-<{name}>.bin'
        return endpoints

    def evals(self, arm, endpoints):
        evals = arm['dir'] / 'evals'
        results = dict(arm=arm['name'], group=arm['group'], program=arm['program'], seed=arm['seed'],
                       purpose=arm['purpose'], init=str(path(arm['init'])), constant_steps=arm['constant_steps'],
                       anneal_steps=arm['anneal_steps'], anneal_floor_lr=arm['anneal_floor_lr'],
                       ema_decay=arm['ema_decay'], ema_half_life=arm['ema_half_life'], flags=arm['flags'], factors=arm.get('factors'),
                       compare=arm.get('compare'), panel=arm['panel'], endpoints={})
        for o, (opponent, spec) in enumerate(arm['evals']):
            games = spec.get('games', arm['games'])
            for e, endpoint in enumerate(arm['endpoint_names']):
                if endpoint not in spec.get('endpoints', arm['endpoint_names']) or endpoint not in endpoints:
                    continue
                stem = f'{endpoint}-{opponent}'
                out, games_out, log_path = (evals / f'{stem}{x}' for x in ('.jsonl', '.games.jsonl', '.log'))
                seed_base = arm['seed_base'] + SEED_STRIDE * (SLOTS * o + e)
                if not self.eval_done(out, games_out, games):
                    shell = f'rm -f {shlex.quote(str(out))}; ' + shlex.join(map(str, [
                        PY, ROOT / 'league/all_combos.py', endpoints[endpoint], path(spec['path']), '--pairs', games // 2,
                        '--workers', arm['eval_workers'], '--seed-base', seed_base,
                        '--label', f"{arm['program']}:{arm['name']}:{endpoint}:{opponent}",
                        '--out', out, '--games-out', games_out]))
                    if not self.dry_run:
                        evals.mkdir(parents=True, exist_ok=True)
                    self.unit(unit_name(arm, f'eval-{stem}'), shell, log_path, 30, '8G',
                              ['OMP_NUM_THREADS=1', f'PYTHONPATH={PUFFER}'])
                    if self.dry_run:
                        continue
                    if not self.eval_done(out, games_out, games):
                        raise RuntimeError(f'eval {out} gave no complete row (see {log_path})')
                row = results['endpoints'].setdefault(endpoint, dict(checkpoint=str(endpoints[endpoint]), opponents={}))
                row['opponents'][opponent] = summary(games_out, seed_base)
        if self.dry_run:
            return
        for row in results['endpoints'].values():
            cells = [row['opponents'][m] for m in arm['panel'] if m in row['opponents']]
            if len(cells) == len(arm['panel']):
                score = sum(c['score'] for c in cells) / len(cells)
                se = math.sqrt(sum(c['se_pairs'] ** 2 for c in cells)) / len(cells)
                row['panel'] = dict(score=score, se=se, ci=[score - 1.96 * se, score + 1.96 * se],
                                    n=sum(c['n'] for c in cells))
        (arm['dir'] / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
        line = ', '.join(f"{ep} {row['panel']['score']:.3f}" for ep, row in results['endpoints'].items() if 'panel' in row)
        log(f"{arm['name']} vs panel: {line}")
        subprocess.run(['curl', '-s', '-m', '10', '-d', f"research {arm['program']}:{arm['name']} vs panel: {line}", NTFY],
                       capture_output=True)

    @staticmethod
    def eval_done(out, games_out, games):
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.startswith('{')] if out.exists() else []
        return bool(rows) and rows[-1]['games'] == games and games_out.exists()


def summary(games_out, seed_base):
    """Score (wins + draws/2), Wilson 95% interval, and the side-swapped pair-clustered standard error."""
    pairs = {}
    for line in Path(games_out).read_text().splitlines():
        g = json.loads(line)
        pairs.setdefault(g['seed'], []).append({'a': 1.0, 'draw': 0.5}.get(g['outcome'], 0.0))
    points = [x for v in pairs.values() for x in v]
    n, p, z = len(points), sum(points) / len(points), 1.959964
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    means = [sum(v) / len(v) for v in pairs.values()]
    m = sum(means) / len(means)
    se = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1) / len(means)) if len(means) > 1 else 0.0
    return dict(score=p, wilson=[centre - half, centre + half], n=n, draws=sum(x == 0.5 for x in points),
                se_pairs=se, seed_base=seed_base, games_out=str(games_out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('configs', nargs='+', type=Path)
    p.add_argument('--dry-run', action='store_true', help='print every command; run nothing')
    p.add_argument('--only', help='comma-separated arm names')
    a = p.parse_args(argv)
    work = []
    for config in a.configs:
        out, arms = load_arms(config)
        runner = Runner(out, a.dry_run)
        work += [(runner, arm) for arm in arms if not a.only or arm['name'] in a.only.split(',')]
    if not work:
        raise SystemExit('no arms selected')
    if not a.dry_run and any(arm['ema_decay'] for _, arm in work):
        check = subprocess.run([PY, '-c', 'from pufferlib import _C; assert hasattr(_C, "save_ema")'],
                               env={'PYTHONPATH': str(PUFFER)}, capture_output=True)
        if check.returncode:
            raise SystemExit(f'{PUFFER} has no EMA support: rebuild with train/build.py --device cuda')
    jobs, failed = queue.Queue(), []

    def evaluate():
        while (job := jobs.get()) is not None:
            runner, arm, endpoints = job
            try:
                runner.evals(arm, endpoints)
            except Exception as error:
                log(f"{arm['name']} eval failed: {error}")
                failed.append(arm['name'])
    evaluator = threading.Thread(target=evaluate)
    evaluator.start()
    try:
        for runner, arm in work:
            try:
                raw = runner.train(arm, 'constant', path(arm['init']))
                annealed = runner.train(arm, 'anneal', raw)
            except Exception as error:
                log(f"{arm['name']} training failed: {error}")
                failed.append(arm['name'])
                continue
            if a.dry_run:
                runner.evals(arm, runner.endpoints(arm, raw, annealed))
            else:
                jobs.put((runner, arm, runner.endpoints(arm, raw, annealed)))
    finally:
        jobs.put(None)
        evaluator.join()
    if failed:
        raise SystemExit(f'failed arms: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
