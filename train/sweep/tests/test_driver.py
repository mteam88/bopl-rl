"""Driver state persistence, resume replay and finalization (stub sweeper)."""
import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import driver as driver_module
import report
import scoring
from test_sweep_config import MINIMAL, load_text

FIXTURE = Path(__file__).resolve().parent / 'fixtures/run'


class StubSweeper:
    """Records observe() calls; suggests fixed values."""

    def __init__(self):
        self.observed = []
        self.suggestion_idx = 0

    def suggest(self, fill, fixed_total_timesteps=None):
        self.suggestion_idx += 1
        fill = dict(fill)
        fill['learning_rate'] = 5e-4
        return fill, {'cost': 1.0}

    def observe(self, hypers, score, cost, is_failure=False):
        self.observed.append((dict(hypers), score, cost, is_failure))


class TestDriver(driver_module.Driver):
    """Stub sweeper; the league scorer plays games, so scoring.curve is stubbed too."""
    curve = [scoring.Point(5001216, 300.0, 100.0), scoring.Point(10002432, 320.0, 200.0)]

    def _make_sweeper(self):
        return StubSweeper()

    def _finalize(self, trial, process):
        with unittest.mock.patch.object(scoring, 'curve', lambda run_dir, cfg: self.curve):
            super()._finalize(trial, process)


def make_driver(tmp, state=None):
    spec = load_text(MINIMAL)
    sweep_dir = Path(tmp) / 'sweep'
    sweep_dir.mkdir(exist_ok=True)
    if state is not None:
        (sweep_dir / 'sweep.json').write_text(json.dumps(state))
    return TestDriver(spec, sweep_dir)


def trial_record(index, status='complete', observations=(), is_failure=False, **extra):
    hypers = dict(steps=2e6, learning_rate=4e-4, head_scale_buttons=1.0, hidden=128)
    return dict(id=f'{index:03d}', index=index, hypers=hypers, seed=73,
                suggested=index > 0, suggestion_info={}, status=status,
                observations=list(observations), is_failure=is_failure,
                early_stop=None, unit=f'bopl-sweep-unit-{index:03d}', **extra)


class Persistence(unittest.TestCase):
    def test_fresh_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            self.assertEqual(d.state['trials'], [])
            d.save()
            reloaded = json.loads((d.sweep_dir / 'sweep.json').read_text())
            self.assertEqual(reloaded['schema'], 1)
            self.assertEqual(reloaded['name'], 'unit')

    def test_replay_reobserves_history_in_order(self):
        observations = [dict(steps=1_000_000, score=0.4, cost=1_000_000, elapsed=60.0),
                        dict(steps=2_000_000, score=0.6, cost=2_000_000, elapsed=120.0)]
        state = dict(schema=1, name='unit', created=0, config={}, trials=[
            trial_record(0, observations=observations),
            trial_record(1, status='failed', is_failure=True),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp, state)
            d.replay()
            self.assertEqual(d.successes, 1)
            self.assertEqual(d.sweeper.suggestion_idx, 1)  # one suggested trial in history
            self.assertEqual(len(d.sweeper.observed), 3)
            (hypers0, score0, cost0, failure0) = d.sweeper.observed[0]
            self.assertEqual((hypers0['steps'], score0, cost0, failure0),
                             (1_000_000, 0.4, 1_000_000, False))
            self.assertEqual(d.sweeper.observed[1][1:], (0.6, 2_000_000, False))
            self.assertTrue(d.sweeper.observed[2][3])  # the failure

    def test_running_trials_skipped_by_replay(self):
        state = dict(schema=1, name='unit', created=0, config={}, trials=[
            trial_record(0, status='running', run_dir='/nonexistent'),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp, state)
            d.replay()
            self.assertEqual(d.sweeper.observed, [])


class Finalize(unittest.TestCase):
    def test_finalize_scores_fixture_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            shutil.copytree(FIXTURE, run_dir)
            d = make_driver(tmp)
            trial = trial_record(0, status='running', run_dir=str(run_dir), launched_at=0)
            d.state['trials'].append(trial)
            d._finalize(trial, None)
            self.assertEqual(trial['status'], 'complete')
            self.assertFalse(trial['is_failure'])
            self.assertEqual(len(trial['observations']), 2)  # two points < downsample 3
            self.assertEqual(trial['observations'][0]['steps'], 5001216)
            self.assertEqual(trial['observations'][0]['cost'], 5001216)  # learner_steps cost
            self.assertEqual(d.successes, 1)
            self.assertEqual(len(d.sweeper.observed), 2)
            # Reports regenerate
            self.assertTrue((d.sweep_dir / 'results.jsonl').exists())
            self.assertTrue((d.sweep_dir / 'summary.md').exists())
            self.assertTrue((d.sweep_dir / 'launch-best.sh').exists())
            # Behaviour flags land in the trial record, the evaluation reports
            # and the run's result.json; the score is unchanged by them.
            self.assertIn('flags', trial)
            self.assertEqual(set(trial['flag_metrics']) >= {'arrows_per_game', 'timeout_rate'}, True)
            stored = json.loads((run_dir / 'evaluations/013336576/result.json').read_text())
            self.assertEqual(stored['flags'], trial['flags'])
            self.assertEqual(json.loads((run_dir / 'result.json').read_text())['flags'], trial['flags'])
            self.assertIn('| flags |', (d.sweep_dir / 'summary.md').read_text())

    def test_flag_rules_from_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'run'
            shutil.copytree(FIXTURE, run_dir)
            d = make_driver(tmp)
            d.spec.flag_rules = {'played': 'games >= 1'}
            trial = trial_record(0, status='running', run_dir=str(run_dir), launched_at=0)
            d.state['trials'].append(trial)
            d._finalize(trial, None)
            self.assertEqual(trial['flags'], ['played'])
            self.assertEqual(trial['observations'][-1]['score'],
                             d.sweeper.observed[-1][1])  # flags do not touch the score
            self.assertIn('| played |', (d.sweep_dir / 'summary.md').read_text())

    def test_finalize_empty_run_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / 'empty'
            run_dir.mkdir()
            d = make_driver(tmp)
            d.curve = []
            trial = trial_record(0, status='running', run_dir=str(run_dir), launched_at=0)
            d.state['trials'].append(trial)
            d._finalize(trial, None)
            self.assertTrue(trial['is_failure'])
            self.assertEqual(trial['status'], 'failed')
            self.assertEqual(d.sweeper.observed, [(trial['hypers'], 0, 0, True)])
            self.assertEqual(d.successes, 0)


class NextTrial(unittest.TestCase):
    def test_first_trial_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            trial = d.next_trial()
            self.assertEqual(trial['id'], '000')
            self.assertFalse(trial['suggested'])
            self.assertEqual(trial['hypers'], d.spec.defaults())
            self.assertEqual(d.sweeper.suggestion_idx, 0)

    def test_later_trials_are_suggestions(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            d.state['trials'].append(trial_record(0))
            trial = d.next_trial()
            self.assertTrue(trial['suggested'])
            self.assertEqual(trial['hypers']['learning_rate'], 5e-4)
            self.assertEqual(sorted(trial['hypers']), sorted(d.spec.params))

    def test_seed_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            self.assertEqual(d.next_trial()['seed'], 73)
            d.spec.seeds = 'increment'
            d.state['trials'].append(trial_record(0))
            self.assertEqual(d.next_trial()['seed'], 74)

    def test_command_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            command = d.command(d.next_trial())
            self.assertEqual(command[0], 'systemd-run')
            self.assertIn('MemoryMax=6G', command)
            self.assertIn('--scope', command)
            self.assertIn('--output', command)
            self.assertIn('--no-eval', command)


class FakeProcess:
    def poll(self):
        return None


class WallCap(unittest.TestCase):
    def test_wall_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp)
            d._run_status = lambda trial: dict(pid=1, learner_steps=1_900_000)
            d.stopped = []
            d.stop_trial = lambda trial, reason: (d.stopped.append(reason),
                                                  trial.__setitem__('early_stop', reason))
            d.spec.max_seconds = 10
            trial = trial_record(0, status='running', run_dir=tmp, launched_at=0)
            self.assertFalse(d.poll(trial, FakeProcess()))
            self.assertEqual(d.stopped, ['wall-cap'])


class Lock(unittest.TestCase):
    def test_live_driver_blocks_second(self):
        state = dict(schema=1, name='unit', created=0, config={}, trials=[],
                     driver=dict(pid=1, started=0))  # init: alive, not ours
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp, state)
            with self.assertRaises(SystemExit):
                d._take_lock()

    def test_same_process_relocks(self):
        import os
        state = dict(schema=1, name='unit', created=0, config={}, trials=[],
                     driver=dict(pid=os.getpid(), started=0))
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp, state)
            d._take_lock()  # our own pid: allowed

    def test_dead_owner_lock_is_taken(self):
        state = dict(schema=1, name='unit', created=0, config={}, trials=[],
                     driver=dict(pid=2 ** 22 - 1, started=0))  # long-dead pid
        with tempfile.TemporaryDirectory() as tmp:
            d = make_driver(tmp, state)
            d._take_lock()
            import os
            self.assertEqual(d.state['driver']['pid'], os.getpid())


class Reports(unittest.TestCase):
    def test_pareto_and_best(self):
        cheap = trial_record(0, observations=[dict(steps=1e6, score=0.5, cost=1e6, elapsed=1)])
        pricey = trial_record(1, observations=[dict(steps=2e6, score=0.7, cost=2e6, elapsed=2)])
        dominated = trial_record(2, observations=[dict(steps=3e6, score=0.6, cost=3e6, elapsed=3)])
        failed = trial_record(3, status='failed', is_failure=True)
        trials = [cheap, pricey, dominated, failed]
        front = report.pareto_front(trials)
        self.assertEqual([t['id'] for t in front], ['000', '001'])
        self.assertEqual(report.best_trial(trials)['id'], '001')

    def test_summary_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = load_text(MINIMAL)
            state = dict(schema=1, trials=[
                trial_record(0, observations=[dict(steps=1e6, score=0.5, cost=1e6, elapsed=1)])])
            report.write_reports(tmp, spec, state, pufferlib_dir='deps/PufferLib')
            summary = (Path(tmp) / 'summary.md').read_text()
            self.assertIn('| 000 |', summary)
            self.assertIn('Pareto front', summary)
            script = (Path(tmp) / 'launch-best.sh').read_text()
            self.assertIn('train.py', script)
            self.assertIn('--learning-rate', script)
            lines = (Path(tmp) / 'results.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)


if __name__ == '__main__':
    unittest.main()
