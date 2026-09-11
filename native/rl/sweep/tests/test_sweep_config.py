"""Config parsing, sweep_config mapping and flag rendering (pure stdlib)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sweep_config

MINIMAL = '''
[sweep]
name = "unit"
method = "Random"
downsample = 3
max_runs = 5

[fixed]
steps = 3000000
workers = 4
banks = 2
entropy = 0.01
entropy_floor = 0.004
entropy_head_scale = "1,1,1"
training_maps = [0, 1]
no_eval = true

[params.steps]
distribution = "log_normal"
min = 1e6
max = 3e6
scale = "time"

[params.learning_rate]
distribution = "log_normal"
min = 1e-5
max = 1e-2
scale = 0.5
default = 3e-4

[params.head_scale_buttons]
distribution = "uniform"
min = 0.5
max = 8.0
scale = "auto"

[params.hidden]
distribution = "uniform_pow2"
min = 32
max = 256
scale = "auto"
default = 128
'''


def load_text(text, name='unit'):
    with tempfile.NamedTemporaryFile('w', suffix='.toml', prefix=name + '-',
                                     delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return sweep_config.load(path)
    finally:
        path.unlink()


class ConfigMapping(unittest.TestCase):
    def test_sweep_config_structure(self):
        spec = load_text(MINIMAL)
        config = spec.sweep_config()
        self.assertEqual(config['method'], 'Random')
        self.assertEqual(config['downsample'], 3)
        self.assertEqual(config['max_runs'], 5)
        self.assertFalse(config['use_gpu'])
        self.assertEqual(config['metric'], 'score')
        self.assertEqual(config['goal'], 'maximize')
        self.assertIn('early_stop_quantile', config)
        self.assertEqual(config['steps'],
                         dict(distribution='log_normal', min=1e6, max=3e6, scale='time'))
        self.assertEqual(config['learning_rate']['distribution'], 'log_normal')
        # `default` stays out of the PufferLib space dict
        self.assertNotIn('default', config['learning_rate'])
        param_names = [k for k in config if isinstance(config[k], dict)]
        self.assertEqual(sorted(param_names),
                         ['head_scale_buttons', 'hidden', 'learning_rate', 'steps'])

    def test_pufferlib_accepts_the_mapping(self):
        try:
            from pufferlib.sweep import Hyperparameters
        except ImportError:
            self.skipTest('pufferlib unavailable outside the training venv')
        spec = load_text(MINIMAL)
        hyperparameters = Hyperparameters(spec.sweep_config(), verbose=False)
        self.assertEqual(hyperparameters.num, 4)
        # Round trip: defaults -> normalized -> dict
        normalized = hyperparameters.from_dict(spec.defaults())
        back = hyperparameters.to_dict(normalized, dict(spec.defaults()))
        self.assertAlmostEqual(back['learning_rate'], 3e-4, places=6)
        self.assertEqual(back['hidden'], 128)

    def test_defaults(self):
        spec = load_text(MINIMAL)
        defaults = spec.defaults()
        self.assertEqual(defaults['steps'], 3000000)     # from [fixed]
        self.assertEqual(defaults['learning_rate'], 3e-4)  # from default =
        self.assertEqual(defaults['head_scale_buttons'], 1.0)  # from the triple
        self.assertEqual(defaults['hidden'], 128)

    def test_default_out_of_range_rejected(self):
        spec = load_text(MINIMAL.replace('default = 3e-4', 'default = 0.5'))
        with self.assertRaises(sweep_config.ConfigError):
            spec.defaults()  # validated when the seed trial is built

    def test_unknown_param_rejected(self):
        with self.assertRaises(sweep_config.ConfigError):
            load_text(MINIMAL + '\n[params.gamma]\ndistribution = "uniform"\nmin = 0.9\nmax = 1.0\n')

    def test_reserved_and_missing(self):
        with self.assertRaises(sweep_config.ConfigError):
            load_text(MINIMAL.replace('[fixed]\nsteps = 3000000', '[fixed]'))
        with self.assertRaises(sweep_config.ConfigError):
            load_text(MINIMAL + '\n[fixed.output]\n')


class FlagRendering(unittest.TestCase):
    def test_trial_arguments(self):
        spec = load_text(MINIMAL)
        hypers = dict(steps=1598242.7, learning_rate=0.000181725,
                      head_scale_buttons=2.5, hidden=64)
        argv = spec.trial_arguments(hypers, Path('/tmp/out'), seed=99)

        def value(flag):
            return argv[argv.index(flag) + 1]

        self.assertEqual(value('--output'), '/tmp/out')
        self.assertEqual(value('--steps'), '1598243')          # integer-rounded
        self.assertEqual(value('--hidden'), '64')
        self.assertEqual(value('--learning-rate'), '0.000181725')
        self.assertEqual(value('--entropy-head-scale'), '1,2.5,1')  # composite
        self.assertEqual(value('--training-maps'), '0,1')      # list join
        self.assertEqual(value('--seed'), '99')
        self.assertIn('--no-eval', argv)                       # store_true
        self.assertNotIn('--head-scale-buttons', argv)         # virtual param

    def test_entropy_floor_capped(self):
        spec = load_text(MINIMAL + '\n[params.entropy]\ndistribution = "log_normal"\n'
                         'min = 1e-3\nmax = 3e-2\nscale = 0.5\n')
        argv = spec.trial_arguments(dict(spec.defaults(), entropy=0.002), Path('/tmp/out'))
        floor = argv[argv.index('--entropy-floor') + 1]
        self.assertEqual(float(floor), 0.001)  # half the swept entropy, below 0.004

    def test_defaults_render_identically_to_fixed(self):
        spec = load_text(MINIMAL)
        argv = spec.trial_arguments(spec.defaults(), Path('/tmp/out'))
        self.assertEqual(argv[argv.index('--steps') + 1], '3000000')
        self.assertEqual(argv[argv.index('--entropy-head-scale') + 1], '1,1,1')


CURRICULUM = '''
[sweep]
name = "curriculum"
method = "Protein"
fixed_cost = true
early_stop = false
max_runs = 3

[fixed]
steps = 200000000
workers = 64
banks = 4
scripted_bots = ["mixed", "archer"]
scripted_level = "hard"
scripted_decay = 100000000
entropy = 0.01
entropy_floor = 0.004
no_eval = true

[params.scripted_banks]
distribution = "int_uniform"
min = 0
max = 2
scale = "auto"

[params.scripted_level_index]
distribution = "int_uniform"
min = 0
max = 2
scale = "auto"

[params.learning_rate]
distribution = "log_normal"
min = 1e-5
max = 1e-2
scale = 0.5
default = 3e-4
'''


class ScriptedCurriculum(unittest.TestCase):
    def test_defaults_from_fixed(self):
        spec = load_text(CURRICULUM)
        self.assertEqual(spec.defaults(), {'scripted_banks': 2, 'scripted_level_index': 2,
                                           'learning_rate': 3e-4})
        self.assertTrue(spec.fixed_cost)
        self.assertFalse(spec.early_stop)

    def test_fixed_cost_adds_a_steps_space(self):
        config = load_text(CURRICULUM).sweep_config()
        self.assertEqual(config['steps']['max'], 200000000)
        self.assertLess(config['steps']['min'], config['steps']['max'])
        import pufferlib.sweep
        pufferlib.sweep.Protein(config, cost_param='steps')

    def test_steps_in_params_rejected_with_fixed_cost(self):
        with self.assertRaises(sweep_config.ConfigError):
            load_text(CURRICULUM + '\n[params.steps]\ndistribution = "log_normal"\nmin = 1e7\nmax = 2e8\nscale = "time"\n')

    def test_virtual_params_render(self):
        spec = load_text(CURRICULUM)
        argv = spec.trial_arguments({'scripted_banks': 1.4, 'scripted_level_index': 0.2,
                                     'learning_rate': 1e-3, 'steps': 12345}, Path('/tmp/x'))
        self.assertEqual(argv[argv.index('--scripted-bots') + 1], 'mixed')
        self.assertEqual(argv[argv.index('--scripted-level') + 1], 'easy')
        self.assertEqual(argv[argv.index('--scripted-decay') + 1], '100000000')
        self.assertEqual(argv[argv.index('--steps') + 1], '200000000')

    def test_no_bots_drops_the_decay(self):
        spec = load_text(CURRICULUM)
        argv = spec.trial_arguments({'scripted_banks': 0, 'scripted_level_index': 2,
                                     'learning_rate': 1e-3}, Path('/tmp/x'))
        self.assertEqual(argv[argv.index('--scripted-bots') + 1], '')
        self.assertNotIn('--scripted-decay', argv)


REFERENCE = MINIMAL + '''
[scoring]
scorer = "reference"
pool = "native/artifacts/rl/sweeps/anchors/overnight-3/pool.json"
last_k = 3
games = 240
band = [0.15, 0.85]

[scoring.flags]
no-offense = "arrows_per_game < 0.05"
stalemate = "timeout_rate > 0.5"
'''


class ReferenceScoring(unittest.TestCase):
    def test_reference_scorer_parses(self):
        spec = load_text(REFERENCE)
        self.assertEqual(spec.scoring['scorer'], 'reference')
        self.assertEqual(spec.scoring['last_k'], 3)
        self.assertEqual(spec.flag_rules, {'no-offense': 'arrows_per_game < 0.05',
                                           'stalemate': 'timeout_rate > 0.5'})

    def test_reference_requires_pool_and_linear_metric(self):
        with self.assertRaises(sweep_config.ConfigError):
            load_text(REFERENCE.replace('pool = "native/artifacts/rl/sweeps/anchors/overnight-3/pool.json"\n', ''))
        with self.assertRaises(sweep_config.ConfigError):
            load_text(REFERENCE.replace('[sweep]\n', '[sweep]\nmetric_distribution = "percentile"\n'))
        with self.assertRaises(sweep_config.ConfigError):
            load_text(REFERENCE.replace('band = [0.15, 0.85]', 'band = [0.9, 0.1]'))

    def test_default_flags_when_absent(self):
        spec = load_text(MINIMAL)
        self.assertEqual(spec.scoring['scorer'], 'anchors')
        self.assertIn('no-offense', spec.flag_rules)

    def test_bad_flag_rule_rejected(self):
        with self.assertRaises(sweep_config.ConfigError):
            load_text(REFERENCE.replace('"timeout_rate > 0.5"', '"movement > 0.5"'))
        with self.assertRaises(sweep_config.ConfigError):
            load_text(REFERENCE.replace('no-offense =', 'No_Offense ='))


if __name__ == '__main__':
    unittest.main()
