"""CPU tests: the trainer's pool runs a PFSP curriculum over the frozen banks.

These drive native_selfplay with a fake backend and synthetic per-bank counters
in exactly the shape static_vec_log emits. That covers the curriculum (who gets
sampled, when a matchup is retired, what the saturation telemetry reports) and
nothing else: the native alignment protocol behind count_aligned, and the rule
that a match whose opponent swapped mid-match is dropped from the per-bank
counters, are C-side behaviours no test in this repo currently exercises.
"""
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import native_selfplay

from league import pfsp


class FakeBackend:
    """Records the calls the pool makes; alignment is scripted per bank."""
    def __init__(self, banks):
        self.loaded = {b: [] for b in range(banks)}
        self.saved = []
        self.aligned = {b: 0 for b in range(banks)}
        self.resets = []

    def save_weights(self, trainer, path):
        Path(path).write_bytes(b'w')
        self.saved.append(path)

    def load_frozen_bank(self, trainer, bank, path):
        self.loaded[bank].append(path)

    def count_aligned(self, trainer, tag, reset):
        value = self.aligned[tag - 1]
        if reset:
            self.resets.append(tag - 1)
            self.aligned[tag - 1] = 0
        return value


def logs(n, **per_bank):
    """Native env log: n-averaged per-bank counts, as static_vec_log emits them.

    Each value is (games, wins, draws); the C side reports games, wins and how
    many games were DECIDED, and native_selfplay derives the draws back out.
    """
    flat = {'env/n': float(n)}
    for bank, (games, wins, draws) in per_bank.items():
        flat[f'env/hist_n_bank_{bank}'] = games / n
        flat[f'env/hist_score_bank_{bank}'] = wins / n
        flat[f'env/hist_decided_bank_{bank}'] = (games - draws) / n
    return flat


class PoolHarness(unittest.TestCase):
    banks = 2

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.pool_dir = Path(self.directory.name)
        self.bootstrap = str(self.pool_dir / 'initial.bin')
        Path(self.bootstrap).write_bytes(b'w')
        self.trainer = types.SimpleNamespace(global_step=0)
        self.backend = FakeBackend(self.banks)
        self.pool = native_selfplay.state(self.pool_dir, self.bootstrap, self.banks, 3, 0, max_size=4,
                                          min_games=4, snapshot_interval=1000, opp_timeout_steps=5000,
                                          pfsp_p=1., seed=1)
        self.ledger = native_selfplay.members(self.bootstrap)

    def tearDown(self):
        self.directory.cleanup()

    def step(self, flat, epoch):
        native_selfplay.credit(self.pool, flat, self.ledger)
        native_selfplay.step(self.trainer, self.backend, self.pool, self.ledger, epoch)
        native_selfplay.register(self.pool, self.ledger, self.trainer.global_step * 4, self.trainer.global_step * 3)

    def summary(self):
        return native_selfplay.summary(self.pool, self.ledger)

    def add_member(self, name, games=0, wins=0, draws=0, permanent=False, kind=None):
        """A pool member with a fixed lifetime record, bypassing play."""
        path = str(self.pool_dir / name)
        Path(path).write_bytes(b'w')
        self.pool['pool'].append(dict(path=path, name=name, permanent=permanent, kind=kind))
        self.ledger[path] = dict(step=None if permanent else len(self.ledger),
                                 learner_steps=None if permanent else 0,
                                 games=games, wins=wins, draws=draws, name=name)
        return path


class DrawAwareCounters(PoolHarness):
    def test_undecided_rounds_are_draws_not_losses(self):
        # 10 games: 2 wins, 8 mutual deaths. The old accumulator scored this
        # 2/10 = 0.2 -- a hard matchup f_hard would pour CPU into. It is 0.6.
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (10, 2, 8), '1': (0, 0, 0)}), 0)
        member = self.ledger[self.bootstrap]
        self.assertEqual((member['games'], member['wins'], member['draws']), (10, 2, 8))
        self.assertAlmostEqual(native_selfplay.scores(self.pool, self.ledger)[self.bootstrap], 0.6)
        bank = self.summary()['banks'][0]
        self.assertAlmostEqual(bank['winrate'], 0.2, msg='the raw win rate is still reported')
        self.assertAlmostEqual(bank['score'], 0.6, msg='the score PFSP uses counts a draw as half')


class OpponentChoice(PoolHarness):
    def draws(self, table, trials=4000):
        """Empirical share of `trials` PFSP draws over a score table."""
        pool = dict(self.pool, pool=[dict(path=name) for name in table], pfsp_p=1.)
        _, f, used_f_var = native_selfplay.intent(pool, table)
        picks = pfsp.sample(pool['rng'], table, f, trials)
        return pfsp.realized(picks, candidates=table), used_f_var

    def test_hopeless_opponent_is_abandoned_under_f_hard(self):
        # We beat `easy` 95-5 and lose to `wall` 2-98, with one real matchup
        # between. f_hard must not spend the pool on the wall.
        shares, used_f_var = self.draws({'easy': 0.95, 'even': 0.5, 'wall': 0.02})
        self.assertFalse(used_f_var, 'we can still beat something, so f_hard applies')
        self.assertGreater(shares['wall'], shares['even'])
        self.assertGreater(shares['even'], shares['easy'])
        # f_hard(x) = 1-x: the mastered opponent all but stops being sampled.
        self.assertAlmostEqual(shares['easy'], 0.05 / 1.53, delta=0.02)

    def test_unbeatable_pool_falls_back_to_the_coin_flip(self):
        # Nothing here is beatable, so f_hard is degenerate: it would send 98%
        # of games to the 100-0 matchup. f_var takes the closest to even.
        shares, used_f_var = self.draws({'wall': 0.01, 'hard': 0.25, 'brutal': 0.1})
        self.assertTrue(used_f_var)
        self.assertGreater(shares['hard'], shares['brutal'])
        self.assertGreater(shares['brutal'], shares['wall'])
        self.assertLess(shares['wall'], 0.05)

    def test_a_freed_slot_draws_a_pfsp_opponent(self):
        # One mastered member and one unplayed one; bank 0 settles on the
        # bootstrap and its replacement must come from the pool, not round-robin.
        mastered = self.add_member('mastered.bin', games=400, wins=400)
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (200, 200, 0)}), 0)
        pending = self.pool['banks'][0]['pending_opp_path']
        self.assertIsNotNone(pending, 'a settled matchup frees the slot')
        self.assertEqual(self.pool['picks'], [pending])
        self.assertIn(pending, [entry['path'] for entry in self.pool['pool']])
        self.assertNotEqual(pending, mastered,
                            'a member we beat 400-0 has weight 0 under f_hard')


class Retirement(PoolHarness):
    def test_a_settled_matchup_is_retired_and_an_open_one_is_not(self):
        # Bank 0: 60 games, 57 wins -- the whole Wilson interval is above 85%.
        # Bank 1: 60 games at an even 50% -- as many games, nothing settled.
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (60, 57, 0), '1': (60, 30, 0)}), 0)
        self.assertIsNotNone(self.pool['banks'][0]['pending_opp_path'])
        self.assertIsNone(self.pool['banks'][1]['pending_opp_path'])

    def test_winning_is_not_enough_while_the_interval_still_straddles_the_band(self):
        # 5 of 6 is an 83% win rate -- over the old --swap-winrate 0.6 -- but
        # the interval runs 44-97%, so the gate buys more games instead.
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (6, 5, 0)}), 0)
        self.assertIsNone(self.pool['banks'][0]['pending_opp_path'])
        self.assertEqual([e['path'] for e in self.pool['pool']], [self.bootstrap],
                         'and takes no snapshot')

    def test_a_matchup_that_settles_against_us_retires_without_a_snapshot(self):
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (60, 3, 0)}), 0)
        self.assertIsNotNone(self.pool['banks'][0]['pending_opp_path'])
        self.assertEqual(len(self.pool['pool']), 1, 'losing 3-57 is not a curriculum rung')

    def test_the_games_floor_holds_a_shutout_open(self):
        self.pool['min_games'] = 100
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (60, 60, 0)}), 0)
        self.assertIsNone(self.pool['banks'][0]['pending_opp_path'])

    def test_the_timeout_frees_a_stalemated_slot(self):
        self.pool['snapshot_interval'] = 0
        self.trainer.global_step = 5000
        before = self.summary()
        self.step(logs(4, **{'0': (10, 5, 0), '1': (10, 5, 0)}), 0)
        after = self.summary()
        self.assertEqual(after['size'], 1, 'a timeout is a stalemate, not progress')
        self.assertTrue(all(bank['pending'] == 'self@0' for bank in after['banks']))
        self.assertEqual([e['event'] for e in native_selfplay.events(before, after, self.ledger, 0)],
                         ['armed', 'armed'])

    def test_a_swap_waits_for_alignment_then_resets_the_matchup(self):
        self.trainer.global_step = 100
        self.step(logs(4, **{'0': (60, 57, 0)}), 0)
        bank = self.pool['banks'][0]
        self.assertEqual(self.backend.loaded[0], [], 'no load before the bank aligned')
        self.backend.aligned[0] = 2
        self.trainer.global_step = 300
        self.step(logs(4), 1)
        self.assertEqual(self.backend.loaded[0], [], 'a partial alignment does not swap')
        self.backend.aligned[0] = 3
        before = self.summary()
        self.trainer.global_step = 400
        self.step(logs(4), 2)
        after = self.summary()
        self.assertEqual(len(self.backend.loaded[0]), 1)
        self.assertEqual(self.backend.resets, [0])
        self.assertEqual((bank['hist_n'], bank['hist_wins'], bank['hist_draws']), (0, 0, 0))
        self.assertEqual(bank['opp_started_step'], 400)
        events = native_selfplay.events(before, after, self.ledger, 1200)
        self.assertEqual([(e['event'], e['bank'], e['epochs_to_align']) for e in events], [('loaded', 0, 2)])
        self.assertEqual(events[0]['retired']['games'], 60)


class PoolGrowth(PoolHarness):
    def test_interval_snapshots_and_eviction(self):
        self.pool['opp_timeout_steps'] = 0
        kinds = []
        for epoch in range(6):
            before = self.summary()
            self.trainer.global_step = 1000 * (epoch + 1)
            self.step(logs(1), epoch)
            after = self.summary()
            kinds += [(e['event'], e['step']) for e in native_selfplay.events(before, after, self.ledger, 0)]
        self.assertEqual(after['size'], 4, 'pool is capped at max_size')
        self.assertEqual([e['step'] for e in after['entries']], [0, 16000, 20000, 24000])
        self.assertEqual(kinds, [('snapshot', 4000), ('snapshot', 8000), ('snapshot', 12000), ('snapshot', 16000),
                                 ('evicted', 4000), ('snapshot', 20000), ('evicted', 8000), ('snapshot', 24000), ('evicted', 12000)])
        self.assertEqual(after['entries'][0]['step'], 0, 'stride eviction never drops the bootstrap anchor')

    def test_banks_settling_together_snapshot_once(self):
        self.trainer.global_step = 100
        before = self.summary()
        self.step(logs(4, **{'0': (60, 57, 0), '1': (60, 57, 0)}), 0)
        after = self.summary()
        paths = [e['path'] for e in self.pool['pool']]
        self.assertEqual(len(paths), len(set(paths)), 'no duplicate pool entries')
        self.assertEqual(after['size'], 2)
        self.assertEqual([e['event'] for e in native_selfplay.events(before, after, self.ledger, 0)],
                         ['snapshot', 'armed', 'armed'])


    def test_prune_deletes_only_evicted_snapshots_no_bank_uses(self):
        # max_size is 4: eleven snapshots leave four candidates (the bootstrap
        # and the newest three) and seven evicted files on disk.
        for step in range(1, 12):
            self.trainer.global_step = step * 1000
            native_selfplay._snapshot(self.trainer, self.backend, self.pool)
        evicted = sorted(set(self.backend.saved) - {e['path'] for e in self.pool['pool']})
        playing, loading = evicted[0], evicted[1]
        self.pool['banks'][0]['cur_opp_path'] = playing
        self.pool['banks'][1]['pending_opp_path'] = loading
        stray = self.pool_dir / 'momentum-000000001.bin'
        stray.write_bytes(b'w')
        deleted = native_selfplay.prune(self.pool)
        self.assertEqual(deleted, evicted[2:])
        self.assertFalse(any(Path(path).exists() for path in deleted))
        for entry in self.pool['pool']:
            self.assertTrue(Path(entry['path']).exists(), f"candidate {entry['name']} kept")
        self.assertTrue(Path(playing).exists() and Path(loading).exists(), 'a file a bank uses waits')
        self.assertTrue(Path(self.bootstrap).exists() and stray.exists(), 'only snapshot files are pruned')
        self.pool['banks'][0]['cur_opp_path'] = self.pool['banks'][1]['pending_opp_path'] = None
        self.assertEqual(native_selfplay.prune(self.pool), [playing, loading])
        self.assertEqual(native_selfplay.prune(self.pool), [])

    def test_prune_matches_files_by_name(self):
        # A resume may spell the run directory differently from the run that
        # recorded the candidates' paths; a live snapshot is still live.
        self.trainer.global_step = 1000
        path = native_selfplay._snapshot(self.trainer, self.backend, self.pool)
        self.pool['pool'][-1]['path'] = '/elsewhere/pool/' + Path(path).name
        self.assertEqual(native_selfplay.prune(self.pool), [])
        self.assertTrue(Path(path).exists())


class Saturation(PoolHarness):
    banks = 4

    def test_a_collapsed_pool_is_visible_in_the_summary(self):
        # Sixteen members, but every game so far went to two of them.
        for i in range(15):
            self.add_member(f'snap{i}.bin')
        hot = [entry['path'] for entry in self.pool['pool'][:2]]
        for path in hot:
            self.ledger[path].update(games=5000, wins=2500, draws=0)
        self.pool['picks'] = [hot[0]] * 30 + [hot[1]] * 30
        saturation = native_selfplay._saturation(self.pool, self.ledger,
                                                 native_selfplay.scores(self.pool, self.ledger))
        self.assertEqual(saturation['played_top2'], 1.0, 'all games went to two opponents')
        self.assertAlmostEqual(saturation['entropy']['played'], 0.6931, places=3)
        self.assertAlmostEqual(saturation['entropy']['realized'], 0.6931, places=3)
        # The 14 unplayed members score 0.5 and the two hot ones 0.5 too, so the
        # pool still *intends* a uniform spread over all sixteen.
        self.assertAlmostEqual(saturation['entropy']['intended'], 2.7726, places=3)
        self.assertGreater(saturation['divergence'], 0.85)

    def test_a_spread_pool_reports_low_divergence(self):
        for i in range(3):
            self.add_member(f'snap{i}.bin', games=100, wins=50)
        self.ledger[self.bootstrap].update(games=100, wins=50)
        self.pool['picks'] = [entry['path'] for entry in self.pool['pool']] * 25
        saturation = native_selfplay._saturation(self.pool, self.ledger,
                                                 native_selfplay.scores(self.pool, self.ledger))
        self.assertAlmostEqual(saturation['divergence'], 0.0, places=6)
        self.assertAlmostEqual(saturation['played_top2'], 0.5)
        self.assertEqual(saturation['candidates'], 4)

    def test_divergence_rises_as_the_realized_draws_concentrate(self):
        for i in range(3):
            self.add_member(f'snap{i}.bin', games=100, wins=50)
        self.ledger[self.bootstrap].update(games=100, wins=50)
        paths = [entry['path'] for entry in self.pool['pool']]
        series = []
        for concentrated in (0, 25, 50, 75, 100):
            self.pool['picks'] = [paths[0]] * concentrated + [p for p in paths for _ in range((100 - concentrated) // 4)]
            series.append(native_selfplay._saturation(
                self.pool, self.ledger, native_selfplay.scores(self.pool, self.ledger))['divergence'])
        self.assertEqual(series, sorted(series), f'divergence must not fall as draws collapse: {series}')
        self.assertGreater(series[-1] - series[0], 0.5)

    def test_an_unsampled_pool_reports_null_not_a_collapsed_one(self):
        # Zero games and zero draws is "unknown", and 0 entropy / 0.5 divergence
        # would read as a saturated pool for the opening stretch of every run.
        fresh = self.summary()['pfsp']
        self.assertIsNone(fresh['divergence'])
        self.assertIsNone(fresh['entropy']['realized'])
        self.assertIsNone(fresh['entropy']['played'])
        self.assertIsNone(fresh['played_top2'])
        self.assertEqual((fresh['draws'], fresh['games']), (0, 0))
        self.assertIsNotNone(fresh['entropy']['intended'], 'the intention is known from the start')

    def test_serving_counts_distinct_opponents_across_banks(self):
        other = self.add_member('other.bin')
        summary = self.summary()
        self.assertEqual(summary['pfsp']['serving'], 1, 'every bank starts on the bootstrap')
        self.pool['banks'][0]['cur_opp_path'] = other
        self.assertEqual(self.summary()['pfsp']['serving'], 2)


class LeagueCandidates(PoolHarness):
    def test_league_members_are_candidates_and_survive_eviction(self):
        # max_size is 4. Fill well past it with snapshots and one league member;
        # the league member must still be a candidate at the end, because it is
        # not this run's history to thin.
        member = self.add_member('search-hard', permanent=True)
        for step in range(1, 12):
            self.trainer.global_step = step * 1000
            native_selfplay._snapshot(self.trainer, self.backend, self.pool)
            native_selfplay.register(self.pool, self.ledger, step, step)
        paths = [e['path'] for e in self.pool['pool']]
        self.assertIn(member, paths, 'a league member is never evicted')
        snapshots = [e for e in self.pool['pool'] if not e['permanent']]
        self.assertLessEqual(len(snapshots), 4, 'snapshots are still bounded by max_size')
        # And it is drawable: PFSP scores it like any other candidate.
        self.assertIn(member, native_selfplay.scores(self.pool, self.ledger))

    def test_telemetry_names_a_league_member(self):
        # A league member has no step, so step keys could not name it.
        self.add_member('search-hard', games=10, wins=2, permanent=True)
        report = self.summary()
        names = {entry['name'] for entry in report['entries']}
        self.assertEqual(names, {'self@0', 'search-hard'})
        self.assertIn('search-hard', report['pfsp']['intended'])
        row = next(e for e in report['entries'] if e['name'] == 'search-hard')
        self.assertIsNone(row['step'], 'a league member is not a snapshot of this run')
        self.assertAlmostEqual(row['score'], 0.2)



class BankGroups(PoolHarness):
    banks = 4

    def settle_bank(self, bank, epoch):
        self.trainer.global_step += 100
        self.step(logs(4, **{str(bank): (200, 200, 0)}), epoch)
        return self.pool['banks'][bank]['pending_opp_path']

    def test_an_exploiter_bank_draws_only_exploiters(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'all', 'exploiters'])
        exploiter = self.add_member('exploiter.bin', games=10, wins=9, permanent=True, kind='exploiter')
        self.add_member('main.bin', games=10, wins=1, permanent=True, kind='main')
        for epoch in range(20):
            pending = self.settle_bank(3, epoch)
            self.assertEqual(pending, exploiter, 'the exploiter bank never draws a non-exploiter')
            self.pool['banks'][3]['pending_opp_path'] = None

    def test_an_exploiter_bank_without_exploiters_draws_the_league(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'all', 'exploiters'])
        main = self.add_member('main.bin', games=10, wins=1, permanent=True, kind='main')
        self.assertEqual(self.settle_bank(3, 0), main)

    def test_a_league_bank_never_draws_the_runs_own_snapshots(self):
        native_selfplay.groups(self.pool, ['league'] * 4)
        target = self.add_member('target.bin', games=10, wins=9, permanent=True, kind='main')
        for epoch in range(20):
            self.assertEqual(self.settle_bank(0, epoch), target)
            self.pool['banks'][0]['pending_opp_path'] = None

    def test_a_bank_off_its_group_swaps_at_once(self):
        # An exploiter's banks start on its own initial weights; they must not
        # play that copy until the opponent timeout before reaching the target.
        native_selfplay.groups(self.pool, ['league'] * 4)
        target = self.add_member('target.bin', permanent=True, kind='main')
        self.trainer.global_step = 1
        self.step(logs(4), 0)
        self.assertEqual([b['pending_opp_path'] for b in self.pool['banks']], [target] * 4)

    def test_a_dropped_member_is_replaced_at_once(self):
        old = self.add_member('old.bin', permanent=True, kind='main')
        self.pool['banks'][0]['cur_opp_path'] = old
        native_selfplay.reconcile(self.pool, self.ledger, [])
        self.trainer.global_step = 1
        self.step(logs(4), 0)
        self.assertNotIn(self.pool['banks'][0]['pending_opp_path'], (None, old))
        self.assertIsNone(self.pool['banks'][1]['pending_opp_path'], 'a bank on a live candidate keeps it')

    def test_a_league_bank_with_no_league_falls_back_to_the_pool(self):
        native_selfplay.groups(self.pool, ['league'] * 4)
        pending = self.settle_bank(0, 0)
        self.assertIn(pending, [e['path'] for e in self.pool['pool'] if not e['permanent']])

    def test_the_whole_pool_bank_still_sees_exploiters(self):
        exploiter = self.add_member('exploiter.bin', permanent=True, kind='exploiter')
        self.assertIn(exploiter, [e['path'] for e in native_selfplay.candidates(self.pool, 'all')])

    def test_a_pinned_bank_draws_only_pinned_members_and_no_other_bank_does(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'league', 'pinned'])
        pinned = self.add_member('camper.bin', games=10, wins=1, permanent=True, kind='exploiter')
        self.pool['pool'][-1]['pinned'] = True
        main = self.add_member('main.bin', games=10, wins=9, permanent=True, kind='main')
        for epoch in range(20):
            self.assertEqual(self.settle_bank(3, epoch), pinned)
            self.pool['banks'][3]['pending_opp_path'] = None
            self.assertEqual(self.settle_bank(2, epoch), main, 'a pinned member is not in the league group')
            self.pool['banks'][2]['pending_opp_path'] = None
        for group in ('all', 'league', 'exploiters'):
            self.assertNotIn(pinned, [e['path'] for e in native_selfplay.candidates(self.pool, group)])

    def test_without_a_pinned_bank_pinned_members_are_ordinary_candidates(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'all', 'exploiters'])
        pinned = self.add_member('camper.bin', permanent=True, kind='exploiter')
        self.pool['pool'][-1]['pinned'] = True
        for group in ('all', 'league', 'exploiters'):
            self.assertIn(pinned, [e['path'] for e in native_selfplay.candidates(self.pool, group)])

    def test_a_forgotten_bank_draws_exploiters_and_the_players_we_lose_to(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'all', 'forgotten'])
        exploiter = self.add_member('exploiter.bin', games=10, wins=9, permanent=True, kind='exploiter')
        lost = self.add_member('old-main.bin', games=10, wins=3, permanent=True, kind='main')
        self.add_member('beaten-main.bin', games=10, wins=8, permanent=True, kind='main')
        table = native_selfplay.scores(self.pool, self.ledger)
        got = {e['path'] for e in native_selfplay.candidates(self.pool, 'forgotten', table)}
        self.assertEqual(got, {exploiter, lost})
        for epoch in range(20):
            self.assertIn(self.settle_bank(3, epoch), (exploiter, lost))
            self.pool['banks'][3]['pending_opp_path'] = None

    def test_a_forgotten_bank_keeps_an_opponent_it_starts_beating(self):
        native_selfplay.groups(self.pool, ['all', 'all', 'all', 'forgotten'])
        self.add_member('exploiter.bin', permanent=True, kind='exploiter')
        main = self.add_member('old-main.bin', games=10, wins=8, permanent=True, kind='main')
        self.pool['banks'][3]['cur_opp_path'] = main
        self.trainer.global_step = 1
        self.step(logs(4), 0)
        self.assertIsNone(self.pool['banks'][3]['pending_opp_path'], 'leaving the group is not staleness')

    def test_a_malformed_spec_is_refused(self):
        for spec in (['all'] * 3, ['all', 'all', 'all', 'mains']):
            with self.assertRaises(ValueError):
                native_selfplay.groups(self.pool, spec)


class Reconcile(PoolHarness):
    def member(self, name, kind='main'):
        path = self.pool_dir / name
        path.write_bytes(b'w')
        return dict(name=name, path=str(path), kind=kind)

    def test_new_members_join_and_unnamed_ones_leave(self):
        old, kept = self.member('old.bin'), self.member('kept.bin')
        added, dropped = native_selfplay.reconcile(self.pool, self.ledger, [old, kept])
        self.assertEqual((sorted(added), dropped), (['kept.bin', 'old.bin'], []))
        self.ledger[kept['path']]['games'] = 7
        new = self.member('new.bin', kind='exploiter')
        added, dropped = native_selfplay.reconcile(self.pool, self.ledger, [kept, new])
        self.assertEqual((added, dropped), (['new.bin'], ['old.bin']))
        paths = [e['path'] for e in self.pool['pool']]
        self.assertIn(self.bootstrap, paths, "the run's own snapshots stay")
        self.assertNotIn(old['path'], paths)
        self.assertEqual(self.ledger[kept['path']]['games'], 7, 'a kept member keeps its record')
        self.assertIn(old['path'], self.ledger, 'a dropped member keeps its record in the ledger')
        self.assertEqual(next(e for e in self.pool['pool'] if e['path'] == new['path'])['kind'], 'exploiter')

    def test_reconciling_the_same_league_is_a_no_op(self):
        league = [self.member('a.bin'), self.member('b.bin')]
        native_selfplay.reconcile(self.pool, self.ledger, league)
        before = [dict(e) for e in self.pool['pool']]
        self.assertEqual(native_selfplay.reconcile(self.pool, self.ledger, league), ([], []))
        self.assertEqual(self.pool['pool'], before)


if __name__ == '__main__':
    unittest.main()
