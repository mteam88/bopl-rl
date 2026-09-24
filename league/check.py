"""Prove the league's estimators on matrices whose answers are known in advance.

This is not a test suite kept for its own sake. Three throwaway measurement
tools in this project have already produced confidently wrong answers -- a
tolerance so loose it would pass any real bug, a probe that zeroed the button it
meant to press, a probe that counted unreachable cases as passes -- and a rating
readout is exactly the kind of tool whose output nobody can eyeball for
correctness. Every check below is a property that, if it broke, would make the
league confidently wrong rather than obviously broken.

It reports how many checks were REACHED, not only how many failed, so a check
that silently stopped running cannot look like a pass.

    python league check
"""
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import pfsp        # noqa: E402
import rating      # noqa: E402
import registry    # noqa: E402
import elo_ladder  # noqa: E402


class Checks:
    def __init__(self):
        self.passed, self.failed, self.skipped = [], [], []

    def ok(self, name, condition, detail=''):
        (self.passed if condition else self.failed).append(f'{name}: {detail}')
        mark = 'pass' if condition else 'FAIL'
        print(f'  [{mark}] {name}' + (f' -- {detail}' if detail else ''))
        return bool(condition)

    def skip(self, name, why):
        self.skipped.append(f'{name}: {why}')
        print(f'  [skip] {name} -- {why}')


def cyclic(size, edge=.7):
    """Score matrix of a size-way rock-paper-scissors: i beats i+1."""
    scores = np.full((size, size), .5)
    for i in range(size):
        scores[i, (i + 1) % size] = edge
        scores[(i + 1) % size, i] = 1 - edge
    np.fill_diagonal(scores, np.nan)
    return scores


def counts_from(names, scores, games=200):
    """{(a, b): [wins_a, wins_b, draws]} realising `scores` exactly, no draws."""
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = names.index(b)
            if np.isnan(scores[i, j]):
                continue
            wins = int(round(scores[i, j] * games))
            out[(a, b) if a < b else (b, a)] = ([wins, games - wins, 0] if a < b
                                                else [games - wins, wins, 0])
    return out


def run():
    check = Checks()
    print('nash / zero-sum equilibria')
    rps = rating.advantage(cyclic(3))
    mixture = rating.maxent_nash(rps)
    check.ok('rps-uniform', np.allclose(mixture, 1 / 3, atol=1e-6),
             f'maxent Nash {np.round(mixture, 4).tolist()}, expected 1/3 each')
    check.ok('rps-value-zero', abs(float(mixture @ rps @ mixture)) < 1e-9,
             f'n.A.n = {float(mixture @ rps @ mixture):+.1e}')

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.dirichlet(np.ones(6))
        arbitrary = rng.normal(size=(6, 6))
        worst = max(worst, abs(float(x @ ((arbitrary - arbitrary.T) / 2) @ x)))
    check.ok('antisymmetry-identity', worst < 1e-12,
             f'max |x.A.x| over 200 random antisymmetric A = {worst:.1e}; a single-population '
             f'"Nash value" is this constant, which is why rate() reports A.n per member instead')

    ladder = rating.advantage(np.array([[np.nan, .7, .9], [.3, np.nan, .7], [.1, .3, np.nan]]))
    top = rating.maxent_nash(ladder)
    check.ok('transitive-collapses', top[0] > .99,
             f'a fully transitive payoff puts {top[0]:.3f} of the mass on the best member')

    # The property Nash is here for, and Elo's failure on the same matrix.
    names = ['A', 'B', 'C']
    base_counts = counts_from(names, cyclic(3))
    cloned = ['A', 'B', 'C', 'C2', 'C3']
    cloned_scores = np.full((5, 5), .5)
    for i in range(5):
        for j in range(5):
            source = (min(i, 2), min(j, 2))
            if i != j:
                cloned_scores[i, j] = .5 if i >= 2 and j >= 2 else cyclic(3)[source]
    np.fill_diagonal(cloned_scores, np.nan)
    base_nash = rating.nash_scores(rating.advantage(cyclic(3)), rating.maxent_nash(rating.advantage(cyclic(3))))
    clone_adv = rating.advantage(cloned_scores)
    clone_nash = rating.nash_scores(clone_adv, rating.maxent_nash(clone_adv))
    drift = float(np.abs(base_nash[:2] - clone_nash[:2]).max())
    check.ok('nash-redundancy-invariant', drift < 1e-6,
             f'cloning a member twice moves the others\' Nash scores by {drift:.1e}')
    base_elo = elo_ladder.bradley_terry_fit(names, base_counts, 'C')['ratings']
    clone_elo = elo_ladder.bradley_terry_fit(cloned, counts_from(cloned, cloned_scores), 'C')['ratings']
    gap_before = abs(base_elo['A'] - base_elo['B'])
    gap_after = abs(clone_elo['A'] - clone_elo['B'])
    check.ok('elo-is-not-invariant', gap_before < 1 and gap_after > 50,
             f'the same clones move the A-B Elo gap {gap_before:.0f} -> {gap_after:.0f}; '
             f'Elo cannot rank a cyclic payoff, which is what a league manufactures')

    print('bradley-terry')
    fit = elo_ladder.bradley_terry_fit(['strong', 'random'], {('strong', 'random'): [750, 250, 0]}, 'random')
    expected = 400 * math.log10(3)
    check.ok('bt-recovers-gap', abs(fit['ratings']['strong'] - expected) < 2,
             f"75% win rate -> {fit['ratings']['strong']:.1f} Elo, closed form {expected:.1f}")
    check.ok('bt-anchors-random', abs(fit['ratings']['random']) < 1e-9,
             f"anchor sits at {fit['ratings']['random']:.1e} Elo")
    draws = elo_ladder.bradley_terry_fit(['a', 'b'], {('a', 'b'): [0, 0, 400]}, 'b')['ratings']
    check.ok('bt-draws-are-half', abs(draws['a']) < 1e-6,
             f'an all-draw pair rates dead level at {draws["a"]:.1e} Elo, not a win for either side')

    print('pfsp')
    scores = {'crushes-us': .05, 'even': .5, 'we-crush': .95}
    hard = pfsp.weights(scores, lambda x: pfsp.f_hard(x, 1.0))
    check.ok('f-hard-direction', hard['crushes-us'] > hard['even'] > hard['we-crush'],
             f"weights {{crushes-us {hard['crushes-us']:.2f}, even {hard['even']:.2f}, "
             f"we-crush {hard['we-crush']:.2f}}}; x is OUR score, so mass goes to who beats us")
    var = pfsp.weights(scores, pfsp.f_var)
    check.ok('f-var-peaks-even', var['even'] > var['crushes-us'] and var['even'] > var['we-crush'],
             f"f_var puts {var['even']:.2f} on the coin-flip")
    check.ok('f-hard-exponent', pfsp.f_hard(.2, 2.) < pfsp.f_hard(.2, 1.) and pfsp.f_hard(.8, 2) < pfsp.f_hard(.8, 1),
             'larger p concentrates harder, as documented in pfsp.py')
    degenerate = pfsp.weights({'a': 1.0, 'b': 1.0}, lambda x: pfsp.f_hard(x, 1.0))
    check.ok('f-hard-all-zero-is-uniform', abs(degenerate['a'] - .5) < 1e-9,
             'beating every candidate 100-0 falls back to uniform rather than dividing by zero')
    picks = pfsp.sample(np.random.default_rng(1), scores, lambda x: pfsp.f_hard(x, 1.0), 40000)
    observed = pfsp.realized(picks, scores)
    check.ok('realized-matches-intended', pfsp.divergence(hard, observed) < .01,
             f'TV distance {pfsp.divergence(hard, observed):.4f} over 40k draws')
    never = pfsp.realized(['a'] * 10, ['a', 'b'])
    check.ok('realized-reports-unplayed', never['b'] == 0.0 and 'b' in never,
             'a candidate that was never drawn appears at 0.000 rather than vanishing')

    print('informative-matchup gating')
    check.ok('gate-drops-decided', not pfsp.informative(48, 0, 0),
             f'48-0-0 has CI {tuple(round(v,3) for v in pfsp.score_interval(48,0,0))}, wholly above 0.85')
    check.ok('gate-keeps-close', pfsp.informative(28, 20, 0), '28-20-0 stays in the schedule')
    check.ok('gate-keeps-unplayed', pfsp.informative(0, 0, 0), 'a pair with no games is informative by definition')
    check.ok('gate-keeps-noisy-lopsided', pfsp.informative(6, 0, 0),
             f'6-0-0 has CI {tuple(round(v,3) for v in pfsp.score_interval(6,0,0))}; too few games to drop')
    # Coverage, computed exactly rather than simulated: for n games there are only
    # n+1 outcomes, so the true coverage of the interval is a finite sum and needs
    # no Monte Carlo (an earlier Monte Carlo version of this check failed at 0.949
    # purely on sampling noise, which is exactly the kind of wobble that gets a
    # real check deleted for being flaky).
    def exact_coverage(n, truth):
        total = 0.0
        for wins in range(n + 1):
            low, high = pfsp.score_interval(wins, n - wins, 0)
            if low <= truth <= high:
                total += math.comb(n, wins) * truth ** wins * (1 - truth) ** (n - wins)
        return total

    grid = [p / 100 for p in range(5, 96, 5)]
    exact = [exact_coverage(48, p) for p in grid]
    check.ok('wilson-coverage-exact', min(exact) >= .90 and sum(exact) / len(exact) >= .94,
             f'over 19 true rates, n=48: min {min(exact):.3f}, mean {sum(exact)/len(exact):.3f} '
             f'(Wilson oscillates about the nominal 0.95 because the outcome is discrete)')
    low, high = pfsp.score_interval(0, 0, 48)
    check.ok('wilson-draws-conservative', low < .5 < high,
             f'48 straight draws score exactly 0.500 with zero variance, yet the interval is '
             f'[{low:.3f}, {high:.3f}] -- treating a draw as half a success overstates the '
             f'spread, so the gate errs toward playing more games, never fewer')

    print('rpp (two populations)')
    dominant = rating.rpp(np.array([[.2, .3], [.1, .15]]))
    check.ok('rpp-saddle', abs(dominant['value'] - .2) < 1e-9,
             f"row 1 dominates, column picks its worst column: v = {dominant['value']:.4f}, expected 0.2000")
    widened = rating.rpp(np.array([[.2, .3, .3], [.1, .15, .15]]))
    check.ok('rpp-composition-invariant', abs(widened['value'] - dominant['value']) < 1e-9,
             f"duplicating an opposing member leaves v at {widened['value']:.4f}")

    print('registry')
    with tempfile.TemporaryDirectory() as directory:
        payoff = Path(directory) / 'payoff.jsonl'
        registry.append([dict(a='x', b='y', condition='c', wins=10, losses=5, draws=3),
                         dict(a='y', b='x', condition='c', wins=2, losses=1, draws=0),
                         dict(a='x', b='y', condition='other', wins=99, losses=0, draws=0)], payoff)
        summed = registry.read_payoff(payoff, condition='c')
        check.ok('payoff-folds-reversed-pairs', summed[('x', 'y')] == [11, 7, 3],
                 f"(x,y) 10-5-3 plus (y,x) 2-1-0 folds to {summed[('x','y')]}, wins and losses swapped")
        check.ok('payoff-never-crosses-conditions', ('x', 'y') in summed and summed[('x', 'y')][0] == 11,
                 "the 99 wins logged under condition 'other' are not summed in")
        check.ok('payoff-conditions-listed', registry.conditions_present(payoff) == {'c': 2, 'other': 1},
                 f'{registry.conditions_present(payoff)}')
        written = [json.loads(line) for line in payoff.read_text().splitlines() if line.strip()]
        check.ok('payoff-written-canonical', all(row['a'] < row['b'] for row in written),
                 'the (y,x) row was stored as (x,y) with wins and losses swapped, so the file '
                 'never holds both orderings of a pair and a consumer may sum without checking direction')
        reversed_row = [r for r in written if r['wins'] == 1][0]
        check.ok('payoff-swap-is-lossless', reversed_row == dict(a='x', b='y', condition='c',
                                                                wins=1, losses=2, draws=0),
                 f'(y,x) 2-1-0 became {reversed_row["a"]},{reversed_row["b"]} 1-2-0')

    # The guard for the failure that actually shipped: `play` wrote member NAMES as
    # payoff keys while every consumer joined on member IDs. It hid because a
    # scripted bot's id IS its name by design, so 10 of 18 participants matched by
    # coincidence and only the 8 checkpoint members diverged -- this CLI reported
    # 153 pairs and 100% coverage while the dashboard reader saw 45 and 7%.
    stored = {key for pair in registry.read_payoff() for key in pair}
    member_ids = {row['id'] for row in registry.load()['members']}
    orphans = sorted(stored - member_ids)
    check.ok('payoff-keys-are-member-ids', not orphans,
             f'{len(stored)} participant keys in payoff.jsonl, all resolve to a member id'
             if not orphans else f'{len(orphans)} keys are not member ids: {orphans[:3]}')
    names = {row['name'] for row in registry.load()['members']}
    ambiguous = stored & (names - member_ids)
    check.ok('payoff-keys-are-not-names', not ambiguous,
             'no participant key is a member name that is not also its id'
             if not ambiguous else f'{len(ambiguous)} keys are names, not ids: {sorted(ambiguous)[:3]}')

    kinds_ok = 'human' in registry.KINDS
    check.ok('human-kind-exists', kinds_ok, f'kinds are {registry.KINDS}')
    aimed = registry.member('e1', 'exploiter', id='x', parent=None, target='dash-bow-platform')
    check.ok('exploiter-carries-target', aimed['target'] == 'dash-bow-platform' and aimed['parent'] is None,
             'target names what it was trained to beat; parent stays null because it began at random init')
    try:
        registry.member('s1', 'scripted', id='s1', target='whatever')
        rejected = False
    except ValueError:
        rejected = True
    check.ok('target-only-where-it-means-something', rejected,
             'a scripted member cannot carry a target')

    league = registry.load()
    verified = league['verified']
    check.ok('league-verifies-on-load', verified['weights_rehashed'] > 0 and not verified['missing'],
             f"{verified['weights_rehashed']} checkpoint weights rehashed, "
             f"{verified['checked']} members checked, missing {verified['missing'] or 'none'}")
    with tempfile.TemporaryDirectory() as directory:
        edited = json.loads(json.dumps(league))
        edited['members'][-1]['name'] = 'tampered'
        path = Path(directory) / 'league.json'
        path.write_text(json.dumps(edited))
        try:
            registry.load(path)
            detected = False
        except ValueError:
            detected = True
        check.ok('league-detects-edits', detected, 'renaming a member after the fact is refused on load')
        swapped = json.loads(json.dumps(league))
        specialists = [m for m in swapped['members'] if m['kind'] == 'specialist']
        if len(specialists) >= 2:
            specialists[0]['path'] = specialists[1]['path']
            swapped['league_sha256'] = registry.league_hash(swapped['members'])
            path.write_text(json.dumps(swapped))
            try:
                registry.load(path)
                caught = False
            except ValueError:
                caught = True
            check.ok('league-detects-weight-swap', caught,
                     'pointing a member at other weights is caught even with the league hash recomputed')
        else:
            check.skip('league-detects-weight-swap', 'fewer than two checkpoint members seeded')

    # Two map sets that differ must not key to the same condition. The original
    # tag concatenated the IDs with no separator, so [1, 12] and [11, 2] both
    # read "112" and their counts would have been pooled.
    check.ok('map-tag-distinguishes-ambiguous-sets',
             registry.map_tag([1, 12]) != registry.map_tag([11, 2]),
             f'{registry.map_tag([1, 12])} vs {registry.map_tag([11, 2])}')
    check.ok('map-tag-names-a-named-set', registry.map_tag(range(124)) == 'mapall-124',
             'a set that is exactly one of map-sets.json\'s stays readable at any size')
    big = registry.map_tag(list(range(40)))
    check.ok('map-tag-stays-short', len(big) <= 12 and big == registry.map_tag(list(reversed(range(40)))),
             f'40 unnamed maps key as {big}, order-independent')
    stored = {json_row['condition'] for json_row in
              (json.loads(line) for line in registry.PAYOFF.read_text().splitlines() if line.strip())}
    check.ok('payoff-conditions-are-registered', not stored - set(league['conditions']),
             f'{len(stored)} condition keys in payoff.jsonl, all registered in league.json'
             if not stored - set(league['conditions'])
             else f'unregistered: {sorted(stored - set(league["conditions"]))}')

    # A condition's fingerprint must not depend on where the league was cloned.
    moved = dict(league['conditions'][sorted(league['conditions'])[0]]['loadouts'])
    here = registry.loadout_sha256(moved)
    moved['set_file'] = '/somewhere/else/' + str(moved.get('set_file') or 'x').rsplit('/', 1)[-1]
    check.ok('condition-fingerprint-is-path-independent', registry.loadout_sha256(moved) == here,
             'the same loadout set measured from another checkout fingerprints the same')

    # Opponent loading. The failure this guards against happened: a run with the
    # `set` encoder loaded eight v4 members into its banks, the 4.4 MB weight
    # files were read into 620 KB policies without complaint, and every one of
    # them played at chance -- the run reported an even score against a member
    # rated 592 Elo above random. Size and shape are not checked by the loader
    # because a bank is a flat float32 array the learner's own kernel indexes.
    loadable = registry.opponents(league)
    shapes = {row['name']: registry.policy_shape(row['path']) for row in loadable}
    check.ok('opponents-are-loadable-members', bool(loadable) and all(row['kind'] != 'scripted' for row in loadable),
             f'{len(loadable)} of {len(league["members"])} members carry weights a bank can hold')
    if loadable:
        mine = dict(shapes[loadable[0]['name']])
        check.ok('opponents-accept-their-own-shape',
                 len(registry.opponents(league, [loadable[0]['name']], policy=mine)) == 1,
                 f"{loadable[0]['name']} loads into a run built the way it was")
        mine['hidden'] = int(mine['hidden']) * 2
        try:
            registry.opponents(league, [loadable[0]['name']], policy=mine)
            refused = False
        except ValueError:
            refused = True
        check.ok('opponents-refuse-another-architecture', refused,
                 'a member from a differently shaped policy cannot be loaded as an opponent')
    try:
        registry.opponents(league, ['not-a-member'])
        named = False
    except ValueError:
        named = True
    check.ok('opponents-reject-unknown-names', named, 'a typo in --opponents fails instead of narrowing the population')

    print('promotion and saturation')
    verdict = rating.promotion('cand', {('cand', 'm1'): [90, 10, 0]}, ['m1', 'm2'])
    check.ok('promotion-counts-unplayed', verdict['admit'] is False and verdict['unplayed'] == ['m2'],
             'a member that was never played is reported unplayed, never as a passed matchup')
    clear = rating.promotion('cand', {('cand', 'm1'): [90, 10, 0]}, ['m1'])
    check.ok('promotion-threshold', clear['admit'] is True,
             f"90-10 clears 70% with interval lower bound {clear['matchups']['m1']['interval'][0]:.3f}")
    narrow = rating.promotion('cand', {('cand', 'm1'): [8, 2, 0]}, ['m1'])
    check.ok('promotion-needs-evidence', narrow['admit'] is False,
             f"8-2 is 80% but its lower bound is {narrow['matchups']['m1']['interval'][0]:.3f}, under 0.70")

    flat = ['random'] + ['p%d' % i for i in range(5)]
    saturated = rating.saturation(flat, counts_from(flat, np.full((6, 6), .5)))
    tiers = np.array([[.5 + .08 * (i - j) for j in range(6)] for i in range(6)])
    spread = rating.saturation(flat, counts_from(flat, tiers))
    missing_anchor = False
    try:
        rating.ratings(['p0', 'p1'], counts_from(['p0', 'p1'], np.array([[np.nan, .7], [.3, np.nan]])))
    except ValueError:
        missing_anchor = True
    check.ok('rating-refuses-without-anchor', missing_anchor,
             'elo_ladder returns all-zero ratings when the anchor is absent; rating.ratings '
             'raises instead of handing back a silent ladder of zeroes')
    check.ok('saturation-separates', saturated['elo_spread'] < spread['elo_spread'],
             f"a pool that is all 0.500 reads Elo spread {saturated['elo_spread']:.1f} vs "
             f"{spread['elo_spread']:.1f} for a tiered one")
    check.ok('saturation-flags-even-pairs', saturated['pairs_near_even'] == 1.0 and spread['pairs_near_even'] < 1.0,
             f"{saturated['pairs_near_even']:.0%} of pairs within 0.45-0.55 when saturated, "
             f"{spread['pairs_near_even']:.0%} when not")
    flipped = {(b, a): [l, w, d] for (a, b), (w, l, d) in counts_from(flat, tiers).items()}
    check.ok('saturation-spread-orientation-invariant',
             abs(spread['pair_score_spread']
                 - rating.saturation(flat, flipped)['pair_score_spread']) < 1e-12,
             f"reading every pair from the other side leaves the spread at "
             f"{spread['pair_score_spread']:.4f}; it measures the pool, not the key order")
    check.ok('saturation-unknown-is-not-zero', saturated['opponent_entropy'] is None,
             'with no realized opponent distribution recorded the field is None, not a confident 0.0')

    total = len(check.passed) + len(check.failed)
    print(f'\n{total} checks reached, {len(check.passed)} passed, {len(check.failed)} failed, '
          f'{len(check.skipped)} skipped')
    for line in check.failed:
        print(f'  FAILED {line}')
    return 1 if check.failed else 0


if __name__ == '__main__':
    raise SystemExit(run())
