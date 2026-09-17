"""Joint ladder: identity, regularization, Wald CI, discovery."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'sweep')]

import elo_ladder
import joint_ladder as jl
import reference_pool as rp


def fake_checkpoint(directory, name, sha):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(bytes.fromhex(sha[:32].ljust(32, '0'))[:16])
    sidecar = json.dumps(dict(schema=1, backend='pufferlib-native', weights_sha256=sha)).encode()
    path.with_name(path.name + '.json').write_bytes(sidecar)
    return sha, path


def fake_pool(directory, name, members, matches, sha_prefix='aa'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    hashed = []
    for member in members:
        row = dict(member)
        if row['kind'] == 'checkpoint' and 'sha256' not in row:
            sha = (sha_prefix + row['name']).ljust(64, '0')[:64]
            fake_checkpoint(directory, row['name'] + '.bin', sha)
            row['path'] = row['name'] + '.bin'
            row['sha256'] = sha
            row['metadata_sha256'] = 'b' * 64
        hashed.append(row)
    pool = dict(schema=rp.SCHEMA, maps=[0, 1, 2], episode_ticks=7200, members=hashed,
                matches=matches, pool_sha256=rp.pool_hash(hashed))
    path = directory / 'pool.json'
    path.write_text(json.dumps(pool))
    loaded = rp.load_pool(path, verify=False)
    loaded['label'] = name
    loaded['sha8'] = rp.pool_sha8(loaded)
    return loaded


def write_rating(run, stem, sha, pool, records, source='band', games=None):
    folder = Path(run) / 'reference' / rp.pool_sha8(pool)
    folder.mkdir(parents=True, exist_ok=True)
    n = games if games is not None else sum(
        row.get('wins', 0) + row.get('losses', 0) + row.get('draws', 0) for row in records.values())
    payload = dict(
        schema=rp.SCHEMA, source=source, checkpoint=stem + '.bin',
        checkpoint_identity=dict(sha256=sha), pool_sha256=pool['pool_sha256'],
        games=n, records=records, rating=0, interval=[0, 0],
    )
    (folder / (stem + '.json')).write_text(json.dumps(payload))
    return payload


class Fit(unittest.TestCase):
    def test_matches_original_on_round_robin(self):
        names = ['random', 'strong', 'weak']
        records = {('random', 'strong'): [8, 40, 0], ('random', 'weak'): [20, 28, 0],
                   ('strong', 'weak'): [36, 10, 2]}
        original = elo_ladder.bradley_terry(names, records, 'random')
        fitted = elo_ladder.bradley_terry_fit(names, records, 'random')
        for name in names:
            self.assertAlmostEqual(original[name], fitted['ratings'][name], delta=0.05)
        self.assertEqual(fitted['ratings']['random'], 0.0)
        self.assertEqual(fitted['intervals']['random'], [0.0, 0.0])
        self.assertLess(fitted['intervals']['weak'][1], fitted['intervals']['strong'][0])

    def test_virtual_draw_does_not_flatten_high_n(self):
        # 480-0 vs random is almost sure; 6-0 is regularized. The 480-game
        # participant must stay well above the 6-game one (if virtual draws
        # scaled with n they would look similar).
        names = ['random', 'many', 'few']
        records = {('many', 'random'): [480, 0, 0], ('few', 'random'): [6, 0, 0]}
        fitted = elo_ladder.bradley_terry_fit(names, records, 'random')
        self.assertGreater(fitted['ratings']['few'], 200)
        self.assertGreater(fitted['ratings']['many'] - fitted['ratings']['few'], 400)
        self.assertTrue(math.isfinite(fitted['ratings']['few']))
        self.assertTrue(math.isfinite(fitted['ratings']['many']))


class Identity(unittest.TestCase):
    def test_pool_name_and_sha_are_one_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sha = 'ab' * 32
            pool_dir = tmp / 'pool-v6'
            members = [
                dict(name='random', kind='random', rating=0.0),
                dict(name='mixed-hard', kind='scripted', rating=250.0),
                dict(name='o7-041-final', kind='checkpoint', rating=600.0, sha256=sha,
                     path='o7-041-final.bin', metadata_sha256='c' * 64),
            ]
            fake_checkpoint(pool_dir, 'o7-041-final.bin', sha)
            matches = [
                dict(a='random', b='mixed-hard', wins_a=10, wins_b=30, draws=0),
                dict(a='random', b='o7-041-final', wins_a=2, wins_b=38, draws=0),
                dict(a='mixed-hard', b='o7-041-final', wins_a=8, wins_b=32, draws=0),
            ]
            pool = fake_pool(pool_dir, 'v6', members, matches)
            # overwrite members with our sha
            pool['members'][2]['sha256'] = sha
            run = tmp / 'sweeps' / 'overnight-7' / '041'
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'metrics.jsonl').write_text('{}\n')
            write_rating(run, 'policy-1333362688', sha, pool, {
                'random': dict(wins=40, losses=2, draws=0, games=42),
                'mixed-hard': dict(wins=32, losses=8, draws=0, games=40),
                'o7-041-final': dict(wins=0, losses=0, draws=0, games=0),
            })
            # A different checkpoint, so the fit has someone else to compare.
            other = 'cd' * 32
            write_rating(run.parent / '000', 'policy-1', other, pool, {
                'random': dict(wins=10, losses=30, draws=0, games=40),
                'mixed-hard': dict(wins=8, losses=32, draws=0, games=40),
            })
            (run.parent / '000' / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (run.parent / '000' / 'metrics.jsonl').write_text('{}\n')
            human = tmp / 'human'
            human.mkdir()
            (human / 'games.jsonl').write_text(json.dumps(dict(
                a='human', b='policy:' + sha, outcome='b', session='s', round=0,
                agent_sha256=sha,
            )) + '\n' + json.dumps(dict(
                a='human', b='bot:mixed-hard', outcome='a', session='s', round=1,
            )) + '\n')
            (human / 'excluded.json').write_text('[]')
            runs = jl.discover_runs(tmp)
            collection = jl.collect(tmp, runs, [pool], human)
            self.assertIn(sha, collection['names'])
            self.assertNotIn('o7-041-final', collection['names'])
            # pool match + candidate records + human all land on the sha node
            vs_random = collection['records'].get((sha, 'random')) or collection['records'].get(('random', sha))
            self.assertIsNotNone(vs_random)
            fitted = jl.fit_collection(collection)
            node = next(row for row in fitted['rows'] if row['id'] == sha)
            self.assertEqual(node['kind'], 'checkpoint')
            self.assertTrue(node['run'] in ('overnight-7-041', 'overnight-7-000') or node['run'])
            human_row = fitted['human']
            self.assertIsNotNone(human_row)
            opponents = {rec['id'] for rec in human_row['records']}
            self.assertIn(sha, opponents)
            self.assertIn('mixed-hard', opponents)

    def test_prefers_band_over_evaluation_same_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sha = '11' * 32
            members = [
                dict(name='random', kind='random', rating=0.0),
                dict(name='mixed-hard', kind='scripted', rating=250.0),
            ]
            pool = fake_pool(tmp / 'pool-v5', 'v5', members, [
                dict(a='random', b='mixed-hard', wins_a=10, wins_b=20, draws=0),
            ])
            run = tmp / 'run'
            (run / 'checkpoints').mkdir(parents=True)
            (run / 'metrics.jsonl').write_text('{}\n')
            write_rating(run, 'policy-1', sha, pool, {
                'random': dict(wins=200, losses=40, draws=0, games=240),
                'mixed-hard': dict(wins=180, losses=60, draws=0, games=240),
            }, source='band', games=480)
            legacy = Path(run) / 'reference' / 'policy-1.json'
            legacy.write_text(json.dumps(dict(
                schema=rp.SCHEMA, source='evaluation', checkpoint='policy-1.bin',
                checkpoint_identity=dict(sha256=sha), pool_sha256=pool['pool_sha256'],
                games=12, records={
                    'random': dict(wins=1, losses=5, draws=0, games=6),
                    'mixed-hard': dict(wins=1, losses=5, draws=0, games=6),
                },
            )))
            eval_only = '22' * 32
            (Path(run) / 'reference' / 'policy-2.json').write_text(json.dumps(dict(
                schema=rp.SCHEMA, source='evaluation', checkpoint='policy-2.bin',
                checkpoint_identity=dict(sha256=eval_only), pool_sha256=pool['pool_sha256'],
                games=12, records={'mixed-hard': dict(wins=6, losses=0, draws=0, games=6)},
            )))
            runs = jl.discover_runs(tmp)
            collection = jl.collect(tmp, runs, [pool], None)
            pair = collection['records'].get((sha, 'random')) or collection['records'].get(('random', sha))
            # band 200-40; the in-run evaluation fit (1-5, no band matching) is not merged in
            self.assertEqual(sum(pair), 240)
            # a checkpoint with only an in-run evaluation fit stays out of the joint fit
            self.assertFalse(any(eval_only in key for key in collection['records']))

    def test_discover_skips_venv_and_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'venv' / 'checkpoints').mkdir(parents=True)
            (tmp / 'sweeps' / 'anchors' / 'pool-v6').mkdir(parents=True)
            (tmp / 'sweeps' / 'anchors' / 'metrics.jsonl').write_text('{}\n')
            real = tmp / 'sweeps' / 'overnight-7' / '041'
            real.mkdir(parents=True)
            (real / 'metrics.jsonl').write_text('{}\n')
            (real / 'checkpoints').mkdir()
            runs = jl.discover_runs(tmp)
            self.assertEqual(set(runs), {'overnight-7-041'})


if __name__ == '__main__':
    unittest.main()
