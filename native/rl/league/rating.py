"""Everything read off the payoff matrix: ratings, Nash, and league health.

Bradley-Terry is NOT reimplemented here. research/elo_ladder.py already fits it
(random anchored at 0 Elo, one virtual draw per played pair, Wald intervals) and
is the fit the rest of the repo uses; this module calls it. What it adds is the
thing Elo structurally cannot do.

Elo assumes strength is one number, so it is only meaningful when the payoff is
close to transitive. Rock-paper-scissors has no Elo ranking, and a league is
built precisely to manufacture non-transitivity. Nash averaging (Balduzzi et
al., "Re-evaluating Evaluation", NeurIPS 2018) instead solves the zero-sum game
whose payoff is the antisymmetric advantage matrix a_ij = w_ij - 1/2 and rates a
member by how it does against the equilibrium mixture. Its point is invariance:
adding redundant copies of a member changes an Elo ladder and does not change a
Nash rating. `check` proves both of those claims on constructed matrices.
"""
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'native/rl/research')]

import elo_ladder  # noqa: E402  (pure until games are played)

ANCHOR = 'random'


# ---- the matrix --------------------------------------------------------------

def matrix(names, counts):
    """(W, N): draw-aware score of row against column, and games played.

    counts is registry.read_payoff output. Unplayed cells are NaN in W and 0 in
    N -- never 0.5, so a caller cannot mistake "never played" for "dead even".
    """
    names = list(names)
    index = {name: i for i, name in enumerate(names)}
    size = len(names)
    wins = np.zeros((size, size))
    games = np.zeros((size, size))
    for (a, b), (wa, la, d) in counts.items():
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        wins[i, j] += wa + d / 2
        wins[j, i] += la + d / 2
        games[i, j] += wa + la + d
        games[j, i] += wa + la + d
    with np.errstate(invalid='ignore', divide='ignore'):
        scores = np.where(games > 0, wins / np.maximum(games, 1), np.nan)
    np.fill_diagonal(scores, np.nan)
    np.fill_diagonal(games, 0)
    return scores, games


def records(names, counts):
    """{(a, b): [wins_a, wins_b, draws]} restricted to `names`, for elo_ladder."""
    keep = set(names)
    return {pair: list(value) for pair, value in counts.items()
            if pair[0] in keep and pair[1] in keep and sum(value) > 0}


def ratings(names, counts, anchor=ANCHOR):
    """Bradley-Terry Elo + Wald 95% intervals, straight from elo_ladder.

    The missing-anchor guard is not defensive noise. elo_ladder.bradley_terry_fit
    returns a dict of all-zero ratings when the anchor is not among `names`, so
    rating a subset that leaves out `random` yields a silent, plausible-looking
    ladder on which every member is exactly 0 Elo and nothing anywhere reports a
    problem. This turns that into a refusal.
    """
    names = list(names)
    if anchor not in names:
        raise ValueError(
            f'anchor {anchor!r} is not among the {len(names)} members being rated; '
            f'Bradley-Terry has no scale without it and would return all zeros')
    return elo_ladder.bradley_terry_fit(names, records(names, counts), anchor)


def bootstrap_ratings(names, counts, anchor=ANCHOR, samples=400, seed=0):
    """Percentile intervals by resampling each pair's counts as a multinomial.

    The Wald interval from the fit assumes the model is right; this one only
    assumes the games in a pair are exchangeable, and is what gets reported when
    a pair has few games.
    """
    rng = np.random.default_rng(seed)
    names = list(names)
    if samples <= 0:
        raise ValueError('bootstrap_ratings needs at least one sample; '
                         'callers wanting no bootstrap should use the fit\'s Wald intervals')
    estimates = {name: [] for name in names}
    base = records(names, counts)
    for _ in range(samples):
        resampled = {}
        for pair, value in base.items():
            total = sum(value)
            resampled[pair] = list(rng.multinomial(total, np.asarray(value, float) / total))
        for name, elo in elo_ladder.bradley_terry(names, resampled, anchor).items():
            estimates[name].append(elo)
    return {name: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for name, v in estimates.items()}


def advantage(scores):
    """a_ij = w_ij - 1/2, antisymmetric, unplayed cells neutral at 0.

    Filling a hole with 0 says "no evidence either way", which is the only
    honest default, but it is still a fill: every readout below reports the
    coverage it was computed at so a confident-looking Nash over a sparse matrix
    is visible as such.
    """
    values = np.where(np.isnan(scores), 0.5, scores) - 0.5
    np.fill_diagonal(values, 0.0)
    return (values - values.T) / 2   # exact antisymmetry despite rounding


# ---- zero-sum equilibria -----------------------------------------------------

def zero_sum_nash(payoff):
    """(x, y, v) for the two-population zero-sum game `payoff` (row maximises).

    Two linear programs, one per player. Used for the genuinely two-population
    case; for one population prefer maxent_nash, which picks a canonical point
    out of what is usually a whole polytope of equilibria.
    """
    from scipy.optimize import linprog
    payoff = np.asarray(payoff, float)
    rows, columns = payoff.shape
    # row: max v s.t. (x^T A)_j >= v for every column j
    upper = np.hstack([-payoff.T, np.ones((columns, 1))])
    result = linprog(c=np.r_[np.zeros(rows), -1.0], A_ub=upper, b_ub=np.zeros(columns),
                     A_eq=np.r_[np.ones(rows), 0.0][None, :], b_eq=[1.0],
                     bounds=[(0, None)] * rows + [(None, None)], method='highs')
    if not result.success:
        raise RuntimeError(f'row LP failed: {result.message}')
    x, v = result.x[:rows], float(result.x[rows])
    # column: min u s.t. (A y)_i <= u for every row i
    upper = np.hstack([payoff, -np.ones((rows, 1))])
    dual = linprog(c=np.r_[np.zeros(columns), 1.0], A_ub=upper, b_ub=np.zeros(rows),
                   A_eq=np.r_[np.ones(columns), 0.0][None, :], b_eq=[1.0],
                   bounds=[(0, None)] * columns + [(None, None)], method='highs')
    if not dual.success:
        raise RuntimeError(f'column LP failed: {dual.message}')
    y, u = dual.x[:columns], float(dual.x[columns])
    if abs(u - v) > 1e-6:
        raise RuntimeError(f'primal {v} and dual {u} disagree; the LP did not converge')
    return x, y, v


def maxent_nash(antisym, tolerance=1e-9):
    """Maximum-entropy Nash of the symmetric zero-sum game `antisym`.

    A symmetric zero-sum game has value 0 and its equilibria form a polytope:
    {x : x >= 0, sum x = 1, (x^T A)_j >= 0 for every j}. Any point in it is a
    Nash, so a bare LP returns an arbitrary vertex and the "rating" would depend
    on which vertex the solver happened to land on. Balduzzi et al. break the
    tie with maximum entropy, the unique choice that spreads mass as evenly as
    the constraints allow and does not privilege an arbitrary member.
    """
    from scipy.optimize import minimize
    antisym = np.asarray(antisym, float)
    size = len(antisym)
    if size == 1:
        return np.ones(1)
    start, _, _ = zero_sum_nash(antisym)
    start = np.clip(start, 1e-6, None)
    start /= start.sum()

    def negative_entropy(x):
        safe = np.clip(x, 1e-12, None)
        return float(np.sum(safe * np.log(safe)))

    def gradient(x):
        return np.log(np.clip(x, 1e-12, None)) + 1.0

    constraints = [dict(type='eq', fun=lambda x: x.sum() - 1.0, jac=lambda x: np.ones_like(x)),
                   dict(type='ineq', fun=lambda x: antisym.T @ x + tolerance,
                        jac=lambda x: antisym.T)]
    found = minimize(negative_entropy, start, jac=gradient, constraints=constraints,
                     bounds=[(0.0, 1.0)] * size, method='SLSQP',
                     options=dict(maxiter=500, ftol=1e-12))
    x = np.clip(found.x, 0.0, None) if found.success else start
    total = x.sum()
    return x / total if total > 0 else start


def nash_scores(antisym, mixture):
    """Each member's expected advantage against the equilibrium mixture, A n.

    This is the per-member Nash rating. It is in score units: +0.10 means the
    member beats the equilibrium mixture 60-40.
    """
    return np.asarray(antisym, float) @ np.asarray(mixture, float)


def rpp(cross):
    """Relative population performance of the row population over the column one.

    v = n1 . A . n2 at the equilibrium of the CROSS-population game (Balduzzi
    et al. 2018). It needs two populations and a rectangular A. For ONE
    population the same expression is identically zero -- A is antisymmetric, so
    x A x^T = 0 for every x, which is just the statement that a symmetric game
    has value 0 -- so a single-population "v" is not a readout, it is a constant.
    `check` asserts that constant rather than reporting it as a finding.
    """
    x, y, v = zero_sum_nash(np.asarray(cross, float))
    return dict(value=float(v), row_mixture=x.tolist(), column_mixture=y.tolist())


# ---- league health -----------------------------------------------------------

def coverage(names, games):
    """How much of the round robin actually exists."""
    size = len(names)
    possible = size * (size - 1) // 2
    upper = np.triu_indices(size, 1)
    played = int((games[upper] > 0).sum())
    per_pair = games[upper][games[upper] > 0]
    return dict(members=size, pairs_possible=possible, pairs_played=played,
                pairs_missing=possible - played,
                fraction=played / possible if possible else 0.0,
                games_total=int(games[upper].sum()),
                games_per_pair_min=int(per_pair.min()) if played else 0,
                games_per_pair_median=float(np.median(per_pair)) if played else 0.0)


def saturation(names, counts, elo=None, mixture=None, opponent_shares=None, newest=None, anchor=ANCHOR):
    """The detector for the failure this project actually had.

    An earlier self-play pool kept only the last four checkpoints. Training win
    rate sat pinned at 0.500 while evaluation against the initial policy climbed
    0.58 -> 0.67, and nothing noticed, because a pool of near-identical recent
    selves is always an even match with itself. Three independent signals, so a
    single blind one cannot hide it:

      spread     -- do members differ at all? Ratings collapsing toward each
                    other, and pair scores collapsing onto 0.500, is the shape
                    of a pool playing itself.
      concentration -- is the realized opponent distribution degenerate? Nash
                    support size, and the entropy of the opponent shares that
                    were actually played (not the weights that were intended).
      ranking_shift -- does a member change anything? Refit without it and
                    compare the order. A member that moves no ranks added no
                    information, whatever its own rating says.
    """
    from scipy.stats import kendalltau
    names = list(names)
    scores, games = matrix(names, counts)
    elo = elo or ratings(names, counts, anchor)['ratings']
    upper = np.triu_indices(len(names), 1)
    pair_scores = scores[upper][games[upper] > 0]
    values = np.array([elo[n] for n in names], float)

    report = dict(
        elo_spread=float(values.std()),
        elo_range=float(values.max() - values.min()) if len(values) else 0.0,
        pair_score_spread=float(pair_scores.std()) if pair_scores.size else 0.0,
        pairs_near_even=float(np.mean(np.abs(pair_scores - .5) < .05)) if pair_scores.size else 0.0,
        pairs_scored=int(pair_scores.size),
    )
    if mixture is not None:
        mixture = np.asarray(mixture, float)
        support = mixture > 1e-3
        report['nash_support'] = int(support.sum())
        report['nash_support_names'] = [n for n, s in zip(names, support) if s]
        report['nash_entropy'] = float(-np.sum(mixture[mixture > 0] * np.log(mixture[mixture > 0])))
        report['nash_entropy_max'] = float(math.log(len(names))) if len(names) > 1 else 0.0
    if opponent_shares:
        shares = np.array([v for v in opponent_shares.values() if v > 0], float)
        report['opponent_entropy'] = float(-np.sum(shares * np.log(shares)))
        report['opponent_entropy_max'] = float(math.log(len(opponent_shares)))
        report['opponent_top_share'] = float(max(opponent_shares.values()))
    else:
        report['opponent_entropy'] = None   # no learner has played yet; not zero, unknown
    if newest and newest in names and newest != anchor and len(names) > 2:
        without = [n for n in names if n != newest]
        reduced = ratings(without, counts, anchor)['ratings']
        full_order = [elo[n] for n in without]
        tau = kendalltau(full_order, [reduced[n] for n in without])
        report['ranking_shift'] = dict(
            member=newest, kendall_tau=float(tau.statistic),
            changed_ranking=bool(tau.statistic < 1 - 1e-9),
            max_elo_move=float(max(abs(elo[n] - reduced[n]) for n in without)))
    return report


def promotion(candidate, counts, against, threshold=0.70):
    """Does `candidate` clear the bar against every member in `against`?

    AlphaStar admitted an exploiter once it beat the agents it was built to
    exploit above 70% (Nature 2019, league composition). The threshold is the
    only part carried over verbatim; because our exploiters restart from random
    weights rather than a fork of the main, "reset" is just "begin a new
    exploiter" and needs no state of its own.

    Reports every matchup with its interval, and says explicitly which members
    were never played -- an unplayed matchup is not a passed one.
    """
    import pfsp
    rows, unplayed = {}, []
    for name in against:
        pair = (candidate, name) if candidate < name else (name, candidate)
        value = counts.get(pair)
        if not value or sum(value) == 0:
            unplayed.append(name)
            continue
        wins, losses, draws = value if candidate < name else [value[1], value[0], value[2]]
        low, high = pfsp.score_interval(wins, losses, draws)
        rows[name] = dict(score=pfsp.score(wins, losses, draws), interval=[low, high],
                          games=wins + losses + draws, clears=low > threshold)
    return dict(candidate=candidate, threshold=threshold, matchups=rows, unplayed=unplayed,
                admit=bool(rows) and not unplayed and all(r['clears'] for r in rows.values()))
