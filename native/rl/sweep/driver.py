"""Protein sweep driver for the Bopl native trainer.

Runs many short train.py trials the way PufferLib's sweep() loop does:
suggest hyperparameters (pufferlib.sweep.Protein / Random / ParetoGenetic),
launch native/rl/puffer/train.py in its own run directory and systemd scope,
score the run with an external yardstick (scoring.py), downsample the score
curve and feed every (cost, score) point back with observe().

    native/artifacts/rl/venv/bin/python native/rl/sweep/driver.py \
        --config native/rl/sweep/configs/<name>.toml

The driver persists every suggestion and observation in
native/artifacts/rl/sweeps/<name>/sweep.json, so it can be killed and
restarted: on start it re-observes the whole history before suggesting again.
--dry-run prints the trial commands without launching or persisting anything.

The first trial always runs the config defaults (PufferLib's "first experiment
uses defaults" rule); Protein's early-stop quantile rule SIGTERMs a running
trial once it falls below the fitted score-at-cost threshold (train.py saves a
checkpoint and exits cleanly on SIGTERM).
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]

# The rift's venv may carry an editable pufferlib install pointing at another
# checkout; resolve PufferLib (and its native _C) inside this tree instead.
PUFFERLIB_DIR = sorted((ROOT / 'native/artifacts/rl/dependencies').glob('PufferLib-*'))[-1]
sys.path.insert(0, str(PUFFERLIB_DIR))

import sweep_config as sweep_config_module
import scoring
import report

VENV_PYTHON = ROOT / 'native/artifacts/rl/venv/bin/python'
TRAIN = ROOT / 'native/rl/puffer/train.py'
DEFAULT_GAME = ROOT / 'sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so'
# pufferl.py only early-stops past min(20% of budget, 100M steps); mirror the
# absolute bound, the fraction comes from [trial].min_early_stop_fraction.
EARLY_STOP_ABSOLUTE_MIN = 100_000_000


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=1) + '\n')
    temporary.replace(path)


def log(message):
    print(f'[sweep {time.strftime("%H:%M:%S")}] {message}', flush=True)


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False


class Driver:
    def __init__(self, spec, sweep_dir, dry_run=0):
        self.spec = spec
        self.sweep_dir = sweep_dir
        self.dry_run = dry_run
        self.state_path = sweep_dir / 'sweep.json'
        self.state = self._load_state()
        self.sweeper = self._make_sweeper()
        self.active = []  # [(trial dict, Popen or None)]
        self.successes = 0
        self._stopping = False

    # ---- construction and persistence ---------------------------------------

    def _load_state(self):
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get('schema') != 1:
                raise SystemExit(f'Unsupported sweep state schema in {self.state_path}')
            return state
        return dict(schema=1, name=self.spec.name, created=time.time(),
                    config=self.spec.raw, trials=[])

    def save(self):
        if not self.dry_run:
            atomic_json(self.state_path, self.state)

    def _make_sweeper(self):
        import pufferlib.sweep
        config = self.spec.sweep_config()
        method = config.pop('method')
        cls = getattr(pufferlib.sweep, method)
        if method == 'Protein':
            sweeper = cls(config, cost_param='steps')
        else:
            sweeper = cls(config)
        return sweeper

    def replay(self):
        """Re-observe persisted history so a restarted driver continues where it stopped."""
        suggested = 0
        for trial in self.state['trials']:
            if trial.get('suggested'):
                suggested += 1
            if trial['status'] == 'running':
                continue  # adopted separately
            self._observe(trial, replay=True)
            if not trial.get('is_failure') and trial.get('observations'):
                self.successes += 1
        if hasattr(self.sweeper, 'suggestion_idx'):
            self.sweeper.suggestion_idx = suggested
        if self.state['trials']:
            log(f"resumed: {len(self.state['trials'])} trials in history, "
                f'{self.successes} successful')

    def adopt_running(self):
        """Trials the previous driver left running: wait for live ones, finalize dead ones."""
        for trial in self.state['trials']:
            if trial['status'] != 'running':
                continue
            status = self._run_status(trial)
            if pid_alive(status.get('pid')):
                log(f"adopting live trial {trial['id']} (pid {status['pid']})")
                self.active.append((trial, None))
            else:
                log(f"finalizing interrupted trial {trial['id']}")
                self._finalize(trial, None)

    # ---- suggestions ---------------------------------------------------------

    def next_trial(self):
        index = len(self.state['trials'])
        if index == 0:
            hypers, info, suggested = self.spec.defaults(), {}, False
        else:
            fill = dict(self.spec.defaults())
            if self.spec.fixed_cost:
                fill['steps'] = int(self.spec.fixed['steps'])
                hypers, info = self.sweeper.suggest(fill, fixed_total_timesteps=fill['steps'])
            else:
                hypers, info = self.sweeper.suggest(fill)
            # Plain Python numbers: numpy scalars subclass float/int but print
            # as np.float64(...) and would look odd in sweep.json.
            hypers = {name: (int(hypers[name]) if self.spec.is_integer(name)
                             else float(hypers[name])) for name in self.spec.params}
            info = {k: (v if isinstance(v, (int, float, str)) else str(v))
                    for k, v in (info or {}).items()}
            suggested = True
        seed = int(self.spec.fixed.get('seed', 73))
        if self.spec.seeds == 'increment':
            seed += index
        trial = dict(id=f'{index:03d}', index=index, hypers=hypers, seed=seed,
                     suggested=suggested, suggestion_info=info, status='pending',
                     observations=[], is_failure=False, early_stop=None,
                     unit=f'{self.spec.unit_prefix}-{self.spec.name}-{index:03d}')
        return trial

    def command(self, trial):
        run_dir = self.sweep_dir / trial['id']
        arguments = self.spec.trial_arguments(trial['hypers'], run_dir, seed=trial['seed'])
        return ['systemd-run', '--user', '--scope', '--collect',
                '-p', f'MemoryMax={self.spec.memory_max}',
                '--unit', trial['unit'], '--',
                'nice', '-n', str(self.spec.nice),
                str(VENV_PYTHON), str(TRAIN), *arguments]

    def environment(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(PUFFERLIB_DIR)
        # Rust picks its own game path and hosts every world in one runtime
        # (train.py); the .NET defaults below would otherwise leak through
        # train.py's setdefault and load one library copy per 4 worlds.
        if env.get('BOPL_RL_BACKEND', '').lower() != 'rust':
            env.setdefault('BOPL_RL_GAME', str(self.spec.fixed.get('game', DEFAULT_GAME)))
            env.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', '4')
        return env

    def launch(self, trial):
        run_dir = self.sweep_dir / trial['id']
        if run_dir.exists():
            raise SystemExit(f'Trial directory {run_dir} already exists; refusing to relaunch')
        log_file = (self.sweep_dir / f"{trial['id']}.log").open('a')
        process = subprocess.Popen(self.command(trial), stdout=log_file,
                                   stderr=subprocess.STDOUT, env=self.environment())
        trial['status'] = 'running'
        trial['launched_at'] = time.time()
        trial['run_dir'] = str(run_dir)
        self.state['trials'].append(trial)
        self.save()
        swept = {k: round(v, 6) if isinstance(v, float) else v for k, v in trial['hypers'].items()}
        log(f"launched trial {trial['id']} ({trial['unit']}): {swept}")
        return process

    # ---- monitoring ----------------------------------------------------------

    def _run_status(self, trial):
        return scoring._read_json(Path(trial.get('run_dir', self.sweep_dir / trial['id'])) / 'status.json', {})

    def stop_trial(self, trial, reason):
        trial['early_stop'] = reason
        log(f"stopping trial {trial['id']} ({reason})")
        subprocess.run(['systemctl', '--user', 'stop', trial['unit'] + '.scope'],
                       capture_output=True)
        status = self._run_status(trial)
        if pid_alive(status.get('pid')):
            try:
                os.kill(status['pid'], signal.SIGTERM)
            except OSError:
                pass

    def budget(self, trial):
        return int(trial['hypers'].get('steps', self.spec.fixed['steps']))

    def cost_of(self, point):
        return point.learner_steps if self.spec.cost == 'learner_steps' else point.elapsed

    def poll(self, trial, process):
        """One monitoring pass. Returns True when the trial has finished."""
        status = self._run_status(trial)
        if process is not None and process.poll() is not None:
            return True
        if process is None and not pid_alive(status.get('pid')):
            return True
        elapsed = time.time() - trial.get('launched_at', time.time())
        if trial['early_stop'] is None and elapsed > self.spec.max_seconds:
            self.stop_trial(trial, 'wall-cap')
            return False
        # Protein quantile early stop against the partial score curve.
        if (trial['early_stop'] is None and self.spec.early_stop and hasattr(self.sweeper, 'should_stop')
                and self.spec.scoring['scorer'] != 'match'):
            learner_steps = status.get('learner_steps', 0)
            floor = min(self.spec.min_early_stop_fraction * self.budget(trial),
                        EARLY_STOP_ABSOLUTE_MIN)
            if learner_steps > floor:
                try:
                    curve = scoring.curve(trial['run_dir'], self.spec.scoring)
                except Exception as error:
                    log(f"scoring poll failed for {trial['id']}: {error}")
                    curve = []
                if curve:
                    point = curve[-1]
                    if self.sweeper.should_stop(point.score, self.cost_of(point)):
                        self.stop_trial(trial, 'quantile')
        return False

    # ---- observation ---------------------------------------------------------

    def _observe(self, trial, replay=False):
        """Feed a finished trial's persisted observations into the sweeper."""
        observations = trial.get('observations') or []
        if trial.get('is_failure') or not observations:
            hypers = dict(trial['hypers'])
            if self.spec.fixed_cost:
                hypers['steps'] = int(self.spec.fixed['steps'])
            self.sweeper.observe(hypers, 0, 0, is_failure=True)
            return
        for entry in observations:
            hypers = dict(trial['hypers'], steps=entry['steps'])
            self.sweeper.observe(hypers, entry['score'], entry['cost'])

    def _finalize(self, trial, process):
        run_dir = Path(trial.get('run_dir', self.sweep_dir / trial['id']))
        result = scoring._read_json(run_dir / 'result.json',
                                    scoring._read_json(run_dir / 'status.json', {}))
        trial['status'] = result.get('status', 'failed')
        trial['finished_at'] = time.time()
        if process is not None:
            trial['exit_code'] = process.returncode
        try:
            curve = scoring.curve(run_dir, self.spec.scoring)
        except Exception as error:
            log(f"scoring failed for {trial['id']}: {error}")
            curve = []
        points = scoring.downsample(curve, self.spec.downsample)
        trial['observations'] = [dict(steps=p.learner_steps, score=p.score,
                                      cost=self.cost_of(p), elapsed=p.elapsed)
                                 for p in points]
        trial['is_failure'] = not trial['observations']
        self._observe(trial)
        if trial['is_failure']:
            log(f"trial {trial['id']} produced no score (status {trial['status']}); "
                'observed as failure')
        else:
            final = trial['observations'][-1]
            self.successes += 1
            log(f"trial {trial['id']} {trial['status']}: score {final['score']:.4f} "
                f"at cost {final['cost']:.4g} ({len(trial['observations'])} observations)")
        self.save()
        if not self.dry_run:
            report.write_reports(self.sweep_dir, self.spec, self.state,
                                 pufferlib_dir=PUFFERLIB_DIR.relative_to(ROOT))

    # ---- main loop -----------------------------------------------------------

    def print_dry_run(self):
        preview = self.dry_run
        log(f'dry run: next {preview} trial commands (nothing launched, nothing persisted)')
        for _ in range(preview):
            trial = self.next_trial()
            self.state['trials'].append(trial)  # in-memory only: advances suggestions
            print(f"\n# trial {trial['id']}"
                  + (' (config defaults)' if not trial['suggested'] else ''))
            print(' '.join(shlex.quote(str(part)) for part in self.command(trial)))

    def _take_lock(self):
        """One driver per sweep: refuse when another live driver owns the state."""
        owner = self.state.get('driver', {})
        if owner.get('pid') and owner['pid'] != os.getpid() and pid_alive(owner['pid']):
            raise SystemExit(f"Another driver (pid {owner['pid']}) is running this sweep; "
                             'stop it first or wait for it to finish')
        self.state['driver'] = dict(pid=os.getpid(), started=time.time())
        self.save()

    def run(self):
        self.replay()
        if self.dry_run:
            self.adopt_running()
            self.print_dry_run()
            return
        self._take_lock()
        self.adopt_running()
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)
        attempts = len(self.state['trials'])
        max_attempts = 2 * self.spec.max_runs
        while not self._stopping:
            while (not self._stopping and len(self.active) < self.spec.concurrency
                    and self.successes + len(self.active) < self.spec.max_runs
                    and attempts < max_attempts):
                trial = self.next_trial()
                process = self.launch(trial)
                attempts += 1
                self.active.append((trial, process))
            if not self.active:
                break
            time.sleep(self.spec.poll_seconds)
            still_active = []
            for trial, process in self.active:
                if self.poll(trial, process):
                    self._finalize(trial, process)
                else:
                    still_active.append((trial, process))
            self.active = still_active
        for trial, process in self.active:
            self.stop_trial(trial, trial.get('early_stop') or 'driver-stop')
            deadline = time.time() + 120
            while time.time() < deadline:
                if self.poll(trial, process):
                    break
                time.sleep(2)
            self._finalize(trial, process)
        self.active = []
        if self.state['trials']:
            report.write_reports(self.sweep_dir, self.spec, self.state,
                                 pufferlib_dir=PUFFERLIB_DIR.relative_to(ROOT))
        log(f"done: {self.successes}/{self.spec.max_runs} successful trials, "
            f"{len(self.state['trials'])} attempts. Reports in {self.sweep_dir}")

    def _request_stop(self, signum, frame):
        if not self._stopping:
            log('stop requested; terminating active trials gracefully')
        self._stopping = True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='Sweep TOML (see configs/)')
    parser.add_argument('--root', type=Path, default=ROOT / 'native/artifacts/rl/sweeps',
                        help='Sweep output root (default native/artifacts/rl/sweeps)')
    parser.add_argument('--dry-run', type=int, nargs='?', const=3, default=0,
                        metavar='N', help='Print the next N trial commands and exit')
    arguments = parser.parse_args()
    spec = sweep_config_module.load(arguments.config)
    sweep_dir = (arguments.root / spec.name).resolve()
    if not arguments.dry_run:
        sweep_dir.mkdir(parents=True, exist_ok=True)
    driver = Driver(spec, sweep_dir, dry_run=arguments.dry_run)
    driver.run()


if __name__ == '__main__':
    main()
