"""Short train.py run of a sweep config: its learner sps, and a pass/fail gate.

    sps_gate.py --config native/rl/sweep/configs/v4-aim-search.toml --seconds 120 --min-sps 120000

Renders the config's default trial exactly as driver.py would (same flags,
same PYTHONPATH / thread environment), runs train.py into --output for
--seconds, stops it with SIGTERM (train.py saves and exits 'stopped'), then
prints the median learner_sps of the last --window updates and exits 0 when it
is at least --min-sps, 3 otherwise (POLICY_V4_PLAN.md 7e: `v4-aim-search` is
dropped from the wave below 120k sps rather than slowed). Overrides
(`--set name=value`) replace [fixed] entries for the gate run only, e.g.
`--set workers=32 --set steps=6000000 --set no_eval=1`.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver          # noqa: E402  (PUFFERLIB_DIR, VENV_PYTHON, TRAIN, Runner.environment)
import sweep_config    # noqa: E402


def parse_override(text):
    name, sep, value = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f'--set {text!r}: expected name=value')
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    return name.strip(), value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='run directory (must not exist)')
    p.add_argument('--seconds', type=float, default=120.0, help='wall time before SIGTERM')
    p.add_argument('--min-sps', type=float, default=0.0, help='gate: median learner_sps of the window must reach this')
    p.add_argument('--window', type=int, default=40, help='last N updates the median is taken over')
    p.add_argument('--set', action='append', default=[], type=parse_override, metavar='NAME=VALUE')
    p.add_argument('--nice', type=int, default=None, help='override [trial].nice for the gate run')
    a = p.parse_args()
    spec = sweep_config.load(a.config)
    for name, value in a.set:
        spec.fixed[name] = value
    hypers = spec.defaults()
    arguments = spec.trial_arguments(hypers, a.output.resolve(), seed=spec.fixed.get('seed'))
    env = dict(os.environ)   # as driver.Runner.environment
    env['PYTHONPATH'] = str(driver.PUFFERLIB_DIR)
    env.setdefault('BOPL_OMP_THREADS', str(os.cpu_count() or 12))
    env.setdefault('OMP_WAIT_POLICY', 'passive')
    env.setdefault('BOPL_ENV_THREADS', str(max(2, (os.cpu_count() or 12) - 4)))
    nice = spec.nice if a.nice is None else a.nice
    command = ['nice', '-n', str(nice), str(driver.VENV_PYTHON), str(driver.TRAIN), *arguments]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log = (a.output.parent / (a.output.name + '.log')).open('w')
    print(f'[gate] {a.config.name}: {a.seconds:.0f} s, then SIGTERM; log {log.name}', flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        process.wait(timeout=a.seconds)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            raise SystemExit('[gate] train.py did not stop within 600 s of SIGTERM')
    log.close()
    elapsed = time.monotonic() - started
    metrics = a.output / 'metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
    if not rows:
        print(open(log.name).read()[-4000:])
        raise SystemExit(f'[gate] no metrics after {elapsed:.0f} s (exit {process.returncode})')
    window = rows[-a.window:]
    sps = statistics.median(r['learner_sps'] for r in window)
    status = json.loads((a.output / 'status.json').read_text()) if (a.output / 'status.json').exists() else {}
    print(f'[gate] {len(rows)} updates in {elapsed:.0f} s, learner_steps {rows[-1]["learner_steps"]}, '
          f'median learner_sps of the last {len(window)}: {sps:.0f} (min {min(r["learner_sps"] for r in window):.0f}, '
          f'max {max(r["learner_sps"] for r in window):.0f}); train.py {status.get("status")} exit {process.returncode}',
          flush=True)
    if a.min_sps and sps < a.min_sps:
        print(f'[gate] FAIL: {sps:.0f} < {a.min_sps:.0f}', flush=True)
        raise SystemExit(3)
    print(f'[gate] PASS' if a.min_sps else '[gate] done', flush=True)


if __name__ == '__main__':
    main()
