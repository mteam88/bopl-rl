"""CPU tests: the trainer's opt-in schedule fixes, resume records and the all_combos kit labels.

native/rl/research/TRAINER-BUGS-2026-10.md lists the bugs these pin. The native
side (pufferlib.cu train_impl reading hypers.prio_beta / hypers.lr_progress, and
bopl_v4.cu's row packing) needs the GPU: train/check_v4.py full_rows and
train/research/check_trainer_fixes_gpu.sh cover it.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<PufferLib> venv/bin/python train/test_trainer_fixes.py
"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'league')]
# Loaded by path: under `python -m unittest train/...` the name `train` is this directory's package.
_spec = importlib.util.spec_from_file_location('bopl_train', HERE / 'train.py')
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

BETA0, ALPHA = 0.2, 0.8


def args(**kw):
    base = dict(steps=1000, schedule_steps=None, prio_beta_fix=0, prio_beta_const=None, anneal_lr_fix=0,
                v4_full_rows='0')
    return types.SimpleNamespace(**{**base, **kw})


def native_beta(epoch, total_epochs):
    """pufferlib.cu train_impl's default beta: the epoch restarts at 0 in every process."""
    return BETA0 + (1 - BETA0) * ALPHA * epoch / total_epochs


def native_cosine(epoch, total_epochs):
    return 0.5 * (1 + math.cos(math.pi * min(1.0, epoch / total_epochs)))


class PrioBeta(unittest.TestCase):
    def test_default_keeps_native(self):
        self.assertEqual(train.prio_beta(args(), BETA0, 500), -1.0)

    def test_native_restarts_at_resume(self):
        # Bug 1 as the native trainer has it: 10 updates of 100 steps, resumed after 5.
        # The resumed process runs epochs 0..4 of a 10-epoch budget again.
        uninterrupted = [native_beta(e, 10) for e in range(10)]
        resumed = [native_beta(e, 5) for e in range(5)] + [native_beta(e, 10) for e in range(5)]
        self.assertNotEqual(uninterrupted, resumed)
        self.assertAlmostEqual(max(uninterrupted), BETA0 + (1 - BETA0) * ALPHA * 0.9)   # never reaches 1

    def test_fix_is_a_function_of_learner_steps(self):
        a = args(prio_beta_fix=1)
        betas = [train.prio_beta(a, BETA0, step) for step in range(0, 1001, 100)]
        self.assertAlmostEqual(betas[0], BETA0)
        self.assertAlmostEqual(betas[-1], 1.0)
        self.assertEqual(betas, sorted(betas))
        # A resumed process sees the restored learner steps, so the schedule is the same.
        self.assertEqual(train.prio_beta(a, BETA0, 500), train.prio_beta(args(prio_beta_fix=1), BETA0, 500))
        self.assertEqual(train.prio_beta(a, BETA0, 5000), 1.0)

    def test_const(self):
        self.assertEqual(train.prio_beta(args(prio_beta_const=0.5), BETA0, 900), 0.5)


class LearningRate(unittest.TestCase):
    def test_default_keeps_native(self):
        self.assertEqual(train.lr_progress(args(), 500), -1.0)

    def test_native_restarts_at_resume(self):
        # Bug 4: a resumed process restarts the cosine at its base value.
        uninterrupted = [native_cosine(e, 10) for e in range(10)]
        resumed = [native_cosine(e, 10) for e in range(5)] + [native_cosine(e, 10) for e in range(5)]
        self.assertEqual(resumed[5], 1.0)
        self.assertLess(uninterrupted[5], 0.6)

    def test_fix_follows_learner_steps(self):
        a = args(anneal_lr_fix=1)
        self.assertEqual(train.lr_progress(a, 0), 0.0)
        self.assertEqual(train.lr_progress(a, 500), 0.5)
        self.assertEqual(train.lr_progress(a, 2000), 1.0)


class ScheduleSteps(unittest.TestCase):
    def test_growing_steps_rewinds_without_it(self):
        before, after = args(anneal_lr_fix=1, prio_beta_fix=1, steps=1000), args(anneal_lr_fix=1, prio_beta_fix=1, steps=2000)
        self.assertGreater(train.lr_progress(before, 1000), train.lr_progress(after, 1000))
        self.assertGreater(train.prio_beta(before, BETA0, 1000), train.prio_beta(after, BETA0, 1000))
        self.assertIn('--schedule-steps', train.schedule_rewinds(after, 1000))

    def test_pinned_horizon_survives_a_grown_budget(self):
        before = args(anneal_lr_fix=1, prio_beta_fix=1, steps=1000, schedule_steps=4000)
        after = args(anneal_lr_fix=1, prio_beta_fix=1, steps=2000, schedule_steps=4000)
        for step in (0, 700, 1000):
            self.assertEqual(train.lr_progress(before, step), train.lr_progress(after, step))
            self.assertEqual(train.prio_beta(before, BETA0, step), train.prio_beta(after, BETA0, step))
        self.assertIsNone(train.schedule_rewinds(after, 1000))

    def test_no_warning_without_fixes_or_growth(self):
        self.assertIsNone(train.schedule_rewinds(args(steps=2000), 1000))
        self.assertIsNone(train.schedule_rewinds(args(anneal_lr_fix=1), 1000))

    def test_recorded_only_when_set(self):
        # Runs recorded before --schedule-steps resume without a spurious 'fixes changed'.
        self.assertEqual(train.schedule_fixes(args()), train.DEFAULT_FIXES)
        self.assertEqual(train.schedule_fixes(args(anneal_lr_fix=1, schedule_steps=7))['schedule_steps'], 7)


class NativeChanges(unittest.TestCase):
    previous = dict(train=dict(gamma=0.995, gae_lambda=0.9, total_timesteps=10, ent_coef=0.01, prio_alpha=0.8),
                    vec=dict(kl_coef=0.0), env=dict(mirror_share_ppm=0))

    def test_unchanged(self):
        current = {k: dict(v) for k, v in self.previous.items()}
        current['train'].update(total_timesteps=20, ent_coef=0.005)   # the budget and the entropy schedule
        self.assertEqual(train.native_changes(self.previous, current), {})

    def test_records_unchecked_hypers(self):
        current = {k: dict(v) for k, v in self.previous.items()}
        current['train'].update(gamma=0.998, prio_alpha=0.0)
        current['env']['mirror_share_ppm'] = 250000
        current['policy'] = dict(hidden_size=256)
        self.assertEqual(train.native_changes(self.previous, current),
                         dict(env=dict(mirror_share_ppm=[0, 250000]), policy=[None, dict(hidden_size=256)],
                              train=dict(gamma=[0.995, 0.998], prio_alpha=[0.8, 0.0])))

    def test_json_normalised(self):
        previous = dict(train=dict(sizes=[1, 2]))
        self.assertEqual(train.native_changes(previous, dict(train=dict(sizes=(1, 2)))), {})


class Overflow(unittest.TestCase):
    def test_accumulates_every_update(self):
        total = train.EMPTY_OVERFLOW
        for packing in ([0, 0, 0, 0, 2, 6, 17, 2, 24576], [0, 0, 3, 0, 2, 6, 20, 2, 24576],
                        [0, 0, 0, 0, 2, 6, 15, 2, 24576, 0, 0, 0, 0], [0, 0, 5, 0, 2, 6, 21, 2, 24576, 0, 0, 4, 0]):
            total = train.accumulate_overflow(total, packing)
        self.assertEqual(total, dict(rows=[0, 0, 8, 0], updates=2, cut=[0, 0, 4, 0]))
        self.assertIs(train.accumulate_overflow(total, None), total)


class Arguments(unittest.TestCase):
    def parse(self, *extra):
        saved = sys.argv
        sys.argv = ['train.py', '--output', '/tmp/x', '--no-eval', *extra]
        try:
            return train.arguments()
        finally:
            sys.argv = saved

    def rejects(self, *extra):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.parse(*extra)

    def test_defaults_are_the_native_behaviour(self):
        a = self.parse()
        self.assertEqual(train.schedule_fixes(a), train.DEFAULT_FIXES)
        self.assertEqual((train.prio_beta(a, BETA0, 10), train.lr_progress(a, 10)), (-1.0, -1.0))

    def test_validation(self):
        self.rejects('--prio-beta-fix', '1', '--prio-beta-const', '0.5')
        self.rejects('--prio-beta-const', '1.5')
        self.rejects('--schedule-steps', '100')                       # needs a fix flag
        self.rejects('--anneal-lr-fix', '1', '--schedule-steps', '0')
        self.rejects('--v4-full-rows', '2')
        self.rejects('--v4-full-rows', '6,12,0,8')
        self.assertEqual(self.parse('--v4-full-rows', '6,12,48,8').v4_full_rows, '6,12,48,8')
        self.assertEqual(self.parse('--anneal-lr-fix', '1', '--schedule-steps', '9').schedule_steps, 9)


class CheckedBackend(unittest.TestCase):
    """pufferl_load_frozen_bank only prints on a bad file and keeps the old weights; save_weights ignores fwrite."""
    class Native:
        def __init__(self, write=16):
            self.loaded, self.write = [], write
        def load_frozen_bank(self, trainer, bank, path):
            self.loaded.append((bank, path))
        def save_weights(self, trainer, path):
            Path(path).write_bytes(b'\0' * self.write)   # a full disk writes short
        def count_aligned(self, trainer, tag, reset):
            return 7

    def test_guards(self):
        import native_selfplay
        with tempfile.TemporaryDirectory() as tmp:
            good, short = Path(tmp, 'good.bin'), Path(tmp, 'short.bin')
            good.write_bytes(b'\0' * 16); short.write_bytes(b'\0' * 8)
            native = self.Native()
            backend = native_selfplay.CheckedBackend(native)
            backend.load_frozen_bank(None, 0, str(short))      # size unknown yet: no check
            backend.nbytes = 16
            backend.load_frozen_bank(None, 1, str(good))
            for path in (short, Path(tmp, 'missing.bin')):
                with self.assertRaises(RuntimeError):
                    backend.load_frozen_bank(None, 2, str(path))
            self.assertEqual([b for b, _ in native.loaded], [0, 1])   # the bad loads never reached the native call
            backend.save_weights(None, str(Path(tmp, 'snap.bin')))
            native.write = 4
            with self.assertRaises(RuntimeError):
                backend.save_weights(None, str(Path(tmp, 'snap2.bin')))
            self.assertEqual(backend.count_aligned(None, 1, 0), 7)


class AllCombosKits(unittest.TestCase):
    """Bug 3 (d5eb7065): A sits in seat `side` (league/games.py), so its kit is kits[3 side: 3 side + 3]."""
    def check(self, tasks):
        self.assertTrue(tasks)
        for task in tasks:
            self.assertEqual(list(task['a_kit']), list(task['kits'][3 * task['side']:3 * task['side'] + 3]), task)
        sides = {t['side'] for t in tasks}
        self.assertEqual(sides, {0, 1})

    def test_all_combos(self):
        import all_combos
        self.check(all_combos.tasks(20, 131001, [0, 1, 2]))
        self.check(all_combos.tasks(6, 131001, [0, 1], b_kits=[[1, 2, 3], [4, 5, 6]]))

    def test_combo_set(self):
        import combo_set
        self.check(combo_set.tasks([(1, 2, 3), (4, 5, 6), (7, 8, 9)], 1, 131001, [0, 1]))


if __name__ == '__main__':
    unittest.main()
