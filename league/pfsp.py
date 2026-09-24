"""Prioritized fictitious self-play: which league member to play next.

AlphaStar samples an opponent B for learner A with probability proportional to
f(P[A beats B]), normalised over the candidates (Vinyals et al., "Grandmaster
level in StarCraft II using multi-agent reinforcement learning", Nature 575
(2019), Supplementary Information, league composition / PFSP pseudocode).

THE EXPONENT CONVENTION, WRITTEN DOWN ON PURPOSE. Two things about `p` are
inconsistent between reimplementations and get silently mixed up:

  1. What x is. Here x is ALWAYS the learner's own win rate against that
     opponent -- its score with draws counted as a half win, in [0, 1]. So
     x near 0 means the opponent crushes us. Some reimplementations pass the
     opponent's win rate instead, which silently inverts the whole curriculum.
  2. What p is. The Nature pseudocode writes f_hard(x) = (1 - x)^p and leaves
     p a hyperparameter. mini-AlphaStar defaults to linear, p = 1; a careful
     independent reimplementation reports weights proportional to
     (1 - x)^(1/2), i.e. p = 1/2, which is gentler and keeps moderately hard
     opponents in the mix.

     We default to p = 1 (linear, the plainest reading of the pseudocode and
     mini-AlphaStar's default). p is an explicit argument on every call so it
     is never implicit in a stored result: larger p concentrates harder on
     opponents that beat us, p = 0 is uniform over candidates.

f_var(x) = x(1-x) ignores direction and concentrates on coin-flips; AlphaStar
uses it where f_hard would be degenerate, i.e. against opponents we cannot beat
at all, where (1-x)^p would pour every game into a matchup we lose 100-0 and
learn nothing from.
"""
import math


def f_hard(x, p=1.0):
    """Weight for an opponent we score x against. See the module docstring for p."""
    return max(0.0, 1.0 - x) ** p


def f_var(x):
    """Weight peaking at x = 0.5: the coin-flips, where the gradient lives."""
    return x * (1.0 - x)


def weights(scores, f):
    """{name: probability} proportional to f(score), normalised.

    scores: {name: our score against them, draws as half a win}.
    When every weight is zero -- we beat the whole candidate set 100-0 under
    f_hard, or every matchup is a shutout under f_var -- there is no preference
    left to express and this falls back to uniform rather than dividing by zero.
    """
    raw = {name: max(0.0, float(f(score))) for name, score in scores.items()}
    total = sum(raw.values())
    if not raw:
        return {}
    if total <= 0:
        return {name: 1.0 / len(raw) for name in raw}
    return {name: value / total for name, value in raw.items()}


def sample(rng, scores, f, count):
    """`count` opponent names drawn i.i.d. from weights(scores, f)."""
    table = weights(scores, f)
    names = sorted(table)
    if not names:
        return []
    probabilities = [table[name] for name in names]
    return [names[i] for i in rng.choice(len(names), size=count, p=probabilities)]


def realized(picks, candidates=()):
    """{name: share} actually drawn, including candidates never picked.

    The intended weights are not the curriculum; what was played is. A pool that
    reported a healthy spread of intended weights while every game in practice
    went to two opponents is what this project's saturation failure looked like
    from the inside, and only the realized distribution shows it.
    """
    counts = {name: 0 for name in candidates}
    for name in picks:
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    return {name: (value / total if total else 0.0) for name, value in counts.items()}


def divergence(intended, observed):
    """Total variation distance between intended weights and realized shares."""
    names = set(intended) | set(observed)
    return 0.5 * sum(abs(intended.get(n, 0.0) - observed.get(n, 0.0)) for n in names)


def entropy(distribution):
    """Shannon entropy in nats; log(n) when every candidate is equally likely."""
    return -sum(p * math.log(p) for p in distribution.values() if p > 0)


# ---- informative-matchup gating ----------------------------------------------

BAND = (0.15, 0.85)


def score(wins, losses, draws):
    """Draw-aware score in [0, 1]: a win is 1, a draw a half, a loss 0."""
    total = wins + losses + draws
    return (wins + draws / 2) / total if total else 0.5


def score_interval(wins, losses, draws, z=1.959963984540054):
    """Wilson interval on the draw-aware score.

    Wilson is a binomial interval and a draw is half a success, which has less
    variance than a coin flip; treating it as one overstates the spread. The
    interval is therefore slightly wider than the truth, which is the safe
    direction for a gate that decides to STOP playing a matchup.
    """
    total = wins + losses + draws
    if total <= 0:
        return 0.0, 1.0
    phat = score(wins, losses, draws)
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def informative(wins, losses, draws, band=BAND):
    """Is this matchup still worth CPU?

    OpenAI Five stopped scheduling matchups more than 10 TrueSkill apart (Dota 2
    paper, appendix); the equivalent here is a score outside 15-85%. A matchup is
    dropped only when its whole confidence interval is outside the band, so a
    lopsided-looking result from a handful of games buys more games rather than
    an early exit. With no games at all it is informative by definition.
    """
    if wins + losses + draws <= 0:
        return True
    low, high = score_interval(wins, losses, draws)
    return not (high < band[0] or low > band[1])
