"""Prioritized fictitious self-play over the trainer's frozen banks.

The banks stay: they are a native construct (backend.load_frozen_bank, perm
tags 1..num_banks, the per-bank counters bopl_puffer.h emits) and playing many
opponents at once is what AlphaStar does too. What changed is which opponent a
freed slot receives, and when a slot is freed.

  - choice: league.pfsp.sample over the current pool, weighted by
    f_hard(x) = (1-x)^p where x is OUR draw-aware score against that member.
    When we cannot beat anything in the pool f_hard is degenerate -- it would
    pour every game into the matchup we lose 100-0 -- so the draw falls back to
    f_var, which peaks on coin-flips. league/pfsp.py owns the conventions.
  - retirement: league.pfsp.informative, a Wilson-interval band gate. A matchup
    is retired when it is SETTLED (its whole confidence interval lies outside
    15-85%), not when we happen to be ahead. The games floor (min_games) and
    the step timeout escape hatch are unchanged.

There is no online Elo here any more. Ratings are league/rating.py's job,
measured post-hoc under one fixed condition; a k=16 Elo updated inside the
training loop conflates every condition it ever saw, which is the thing the
league registry exists to avoid.

Draws are real. bopl_puffer.h emits hist_decided_bank_{b} beside
hist_n_bank_{b} / hist_score_bank_{b}, so wins/losses/draws are derived exactly
instead of charging every undecided round as a loss. That mattered: f_hard
would have read a draw-heavy matchup as a hard one and poured CPU into it.

Swaps replace a bank's weights in place while its matches run (the alignment
gate only waits until every environment of the bank has started a match since
the previous swap). The environment therefore drops a match whose opponent
changed mid-match from the pool's per-bank statistics (see bopl_puffer.h).
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from league import pfsp
from pufferlib.selfplay import evict

# AlphaStar switches a main agent from f_hard to f_var when it cannot beat the
# candidates (win rate under ~30%). Read over the whole set: when our best
# score against any pool member is below this, every f_hard weight sits on a
# matchup we learn nothing from, so the draw chases coin-flips instead.
HOPELESS = 0.3

_NEW = dict(step=None, learner_steps=None, games=0, wins=0, draws=0, name='?')
# A run snapshot's candidate name. League members bring their own.
SELF = 'self@%d'


def _thin(entries, max_size):
    """evict() over this run's snapshots; league members are never dropped."""
    snapshots = [e for e in entries if not e.get('permanent')]
    kept = {id(e) for e in evict(snapshots, max_size)}
    return [e for e in entries if e.get('permanent') or id(e) in kept]


def state(pool_dir, bootstrap, banks, historical_per_bank, global_step, *, max_size,
          min_games, snapshot_interval, opp_timeout_steps, pfsp_p, seed, league=()):
    """`league` is the committed members this run plays besides its own history.

    Each is dict(name, path). They are PERMANENT candidates: eviction thins the
    run's own snapshots and never touches them, because they are not this run's
    history to forget -- they are the standing population the run is measured
    against, and dropping one would silently narrow the curriculum. max_size
    therefore bounds the snapshots, not the candidate set.
    """
    if banks < 1 or historical_per_bank < 1 or max_size < 1 or min_games < 1:
        raise ValueError('The pool needs at least one bank, historical game, entry and game per swap')
    if not pfsp_p >= 0:
        raise ValueError('pfsp_p must be non-negative (0 is uniform over candidates)')
    league = [dict(path=str(m['path']), name=str(m['name']), permanent=True) for m in league]
    if len({m['name'] for m in league}) != len(league):
        raise ValueError('League candidates must have distinct names')
    bank = dict(cur_opp_path=bootstrap, hist_n=0, hist_wins=0, hist_draws=0,
                pending_opp_path=None, epoch_armed=0, opp_started_step=global_step,
                num_hist_envs=historical_per_bank, last_score_at_swap=0.0, last_epochs_to_align=0)
    return dict(pool_dir=str(pool_dir),
                pool=[dict(path=bootstrap, name=SELF % 0, permanent=False)] + league,
                rng=np.random.default_rng(seed), max_size=max_size, min_games=min_games,
                snapshot_interval=snapshot_interval, opp_timeout_steps=opp_timeout_steps,
                num_banks=banks, banks=[dict(bank) for _ in range(banks)], pfsp_p=float(pfsp_p),
                # Every opponent this pool ever drew, in order. pfsp.realized over
                # this is the only honest answer to "what curriculum did the
                # learner get"; the intended weights are a wish, not a record.
                picks=[], last_snapshot_step=global_step)


def members(bootstrap, league=()):
    """Ledger of every candidate the run ever held: age and lifetime record against the learner.

    A league member has no step: it is not a snapshot of this run, and reporting
    0 would put it at the origin of the run's own history.
    """
    ledger = {bootstrap: dict(step=0, learner_steps=0, games=0, wins=0, draws=0, name=SELF % 0)}
    for m in league:
        ledger[str(m['path'])] = dict(step=None, learner_steps=None, games=0, wins=0, draws=0,
                                      name=str(m['name']))
    return ledger


def _wld(games, wins, draws):
    return wins, games - wins - draws, draws


def _entry(ledger, path):
    """A ledger row, tolerating a snapshot taken this update that register() has not seen."""
    return ledger.get(path, _NEW)


def credit(pool, flat, ledger):
    """Attribute the window's finished games to each bank's current opponent.

    Call before step(): it scores the same games, then may retire the matchup.
    """
    n = float(flat.get('env/n', 0.0))
    for index, bank in enumerate(pool['banks']):
        games = round(float(flat.get(f'env/hist_n_bank_{index}', 0.0)) * n)
        if not games:
            continue
        # The counters arrive divided by n and are multiplied back out, so each
        # rounds independently; clamp rather than let a one-game rounding
        # disagreement reach pfsp.score_interval as a negative loss count.
        wins = min(games, max(0, round(float(flat.get(f'env/hist_score_bank_{index}', 0.0)) * n)))
        # Undecided rounds and timeouts are draws in the opponent statistic.
        # What they PAY the agent is a separate, deliberate decision (-1).
        decided = min(games, max(wins, round(float(flat.get(f'env/hist_decided_bank_{index}', 0.0)) * n)))
        bank['hist_n'] += games
        bank['hist_wins'] += wins
        bank['hist_draws'] += games - decided
        member = ledger[bank['cur_opp_path']]
        member['games'] += games
        member['wins'] += wins
        member['draws'] += games - decided


def scores(pool, ledger):
    """Our draw-aware score against every current pool member, in [0, 1].

    A member we have never played scores 0.5 (pfsp.score's no-games answer):
    unknown, which is neither hopeless nor mastered.
    """
    table = {}
    for entry in pool['pool']:
        row = _entry(ledger, entry['path'])
        table[entry['path']] = pfsp.score(*_wld(row['games'], row['wins'], row['draws']))
    return table


def intent(pool, table):
    """(weights, f, used_f_var) for the current candidate set."""
    hopeless = bool(table) and max(table.values()) < HOPELESS
    f = pfsp.f_var if hopeless else (lambda x, p=pool['pfsp_p']: pfsp.f_hard(x, p))
    return pfsp.weights(table, f), f, hopeless


def _snapshot(trainer, backend, pool):
    """Save the learner as a new pool member. Idempotent within one update."""
    path = os.path.join(pool['pool_dir'], f'{int(trainer.global_step):016d}.bin')
    if not any(entry['path'] == path for entry in pool['pool']):
        backend.save_weights(trainer, path)
        pool['pool'].append(dict(path=path, name=SELF % int(trainer.global_step), permanent=False))
        pool['pool'] = _thin(pool['pool'], pool['max_size'])
    pool['last_snapshot_step'] = int(trainer.global_step)
    return path


def step(trainer, backend, pool, ledger, epoch):
    """One update of pool bookkeeping: snapshots, retirements, PFSP draws, telemetry."""
    global_step = int(trainer.global_step)

    if (pool['snapshot_interval'] > 0
            and global_step - pool['last_snapshot_step'] >= pool['snapshot_interval']):
        _snapshot(trainer, backend, pool)

    table = scores(pool, ledger)
    for index, bank in enumerate(pool['banks']):
        tag = index + 1
        if bank['pending_opp_path'] is not None:
            if backend.count_aligned(trainer, tag, 0) >= bank['num_hist_envs']:
                backend.load_frozen_bank(trainer, index, bank['pending_opp_path'])
                backend.count_aligned(trainer, tag, 1)
                bank['cur_opp_path'] = bank['pending_opp_path']
                bank['pending_opp_path'] = None
                bank['hist_n'] = bank['hist_wins'] = bank['hist_draws'] = 0
                bank['opp_started_step'] = global_step
                bank['last_epochs_to_align'] = epoch - bank['epoch_armed']
            continue

        wins, losses, draws = _wld(bank['hist_n'], bank['hist_wins'], bank['hist_draws'])
        settled = bank['hist_n'] >= pool['min_games'] and not pfsp.informative(wins, losses, draws)
        timed_out = (pool['opp_timeout_steps'] > 0
                     and global_step - bank['opp_started_step'] >= pool['opp_timeout_steps'])
        if not (settled or timed_out):
            continue
        # The pool grows when the learner OUTGREW a matchup -- the whole
        # interval is above the band -- because that is a new curriculum rung.
        # Settling the other way, or merely timing out, is not progress.
        if settled and pfsp.score_interval(wins, losses, draws)[0] > pfsp.BAND[1]:
            _snapshot(trainer, backend, pool)
            table = scores(pool, ledger)
        _, f, _ = intent(pool, table)
        path, = pfsp.sample(pool['rng'], table, f, 1)
        pool['picks'].append(path)
        bank['pending_opp_path'] = path
        bank['epoch_armed'] = epoch
        bank['last_score_at_swap'] = pfsp.score(wins, losses, draws)


def _saturation(pool, ledger, table):
    """The four distributions a human needs to see a pool collapsing."""
    paths = [entry['path'] for entry in pool['pool']]
    weights, _, used_f_var = intent(pool, table)
    observed = pfsp.realized(pool['picks'], candidates=paths)
    total = sum(row['games'] for row in ledger.values())
    played = {path: (row['games'] / total if total else 0.0) for path, row in ledger.items()}
    ranked = sorted(played.values(), reverse=True)
    # Before the first draw, and before the first game, the realized
    # distributions are empty rather than collapsed. Reporting 0 entropy and a
    # 0.5 divergence there would read as a saturated pool for the opening
    # stretch of every run, so those are null until there is a sample.
    drawn, staked = bool(pool['picks']), bool(total)
    return dict(p=pool['pfsp_p'], f='var' if used_f_var else 'hard', candidates=len(paths),
                draws=len(pool['picks']), games=total,
                intended=weights, realized=observed, played=played,
                # Intended is what the pool wants NOW; realized is every draw it
                # ever made. A pool whose intention keeps saying "spread" while
                # the draws keep landing on two members holds this high.
                divergence=pfsp.divergence(weights, observed) if drawn else None,
                entropy=dict(intended=pfsp.entropy(weights),
                             realized=pfsp.entropy(observed) if drawn else None,
                             played=pfsp.entropy(played) if staked else None),
                # The blunt collapse signature: games share of the top two
                # opponents, and how many distinct opponents the banks serve.
                played_top2=sum(ranked[:2]) if staked else None,
                serving=len({bank['cur_opp_path'] for bank in pool['banks']}))


def register(pool, ledger, step, learner_steps):
    """Record when snapshots step() just took were made."""
    for entry in pool['pool']:
        ledger.setdefault(entry['path'], dict(step=step, learner_steps=learner_steps,
                                              games=0, wins=0, draws=0, name=entry['name']))


def _member(ledger, path):
    row = ledger[path]
    wins, losses, draws = _wld(row['games'], row['wins'], row['draws'])
    return dict(name=row['name'], step=row['step'], learner_steps=row['learner_steps'],
                games=row['games'],
                winrate=row['wins'] / row['games'] if row['games'] else None,
                score=pfsp.score(wins, losses, draws) if row['games'] else None)


def summary(pool, ledger, saturation=True):
    """The pool's PFSP state for metrics.jsonl.

    saturation=False leaves out the 'pfsp' block, most of the cost: the
    update loop's pre-step summary feeds only events().
    """
    banks = []
    for bank in pool['banks']:
        games = bank['hist_n']
        wins, losses, draws = _wld(games, bank['hist_wins'], bank['hist_draws'])
        banks.append(dict(opponent=ledger[bank['cur_opp_path']]['name'], games=games,
                          winrate=bank['hist_wins'] / games if games else None,
                          score=pfsp.score(wins, losses, draws) if games else None,
                          draws=draws,
                          pending=_entry(ledger, bank['pending_opp_path'])['name']
                          if bank['pending_opp_path'] else None,
                          score_at_swap=bank['last_score_at_swap'],
                          epochs_to_align=bank['last_epochs_to_align']))
    games = sum(bank['hist_n'] for bank in pool['banks'])
    wins = sum(bank['hist_wins'] for bank in pool['banks'])
    out = dict(schema=2, size=len(pool['pool']), entries=[_member(ledger, entry['path']) for entry in pool['pool']],
               winrate=wins / games if games else None, banks=banks)
    if not saturation:
        return out
    saturation = _saturation(pool, ledger, scores(pool, ledger))
    # Keyed by candidate NAME, not by step: a league member is not a snapshot of
    # this run and has no step, so a step key could not name one.
    name_of = {path: ledger[path]['name'] for path in ledger}
    for key in ('intended', 'realized'):
        saturation[key] = {name_of.get(path, '?'): share for path, share in saturation[key].items()}
    # Members the pool has evicted still consumed curriculum; keep them as one
    # number rather than growing every metrics line with the whole ledger.
    live = {entry['path'] for entry in pool['pool']}
    played = saturation.pop('played')
    saturation['played'] = {name_of[path]: share for path, share in played.items() if path in live}
    saturation['played_evicted'] = sum(share for path, share in played.items() if path not in live)
    out['pfsp'] = saturation
    return out


def events(before, after, ledger, learner_steps):
    """pool.jsonl records for what step() changed between two summaries."""
    out = []
    old = {entry['name']: entry for entry in before['entries']}
    new = {entry['name']: entry for entry in after['entries']}
    for name in sorted(new.keys() - old.keys()):
        out.append(dict(event='snapshot', name=name, step=new[name]['step']))
    for name in sorted(old.keys() - new.keys()):
        out.append(dict(event='evicted', **old[name]))
    for index, (was, now) in enumerate(zip(before['banks'], after['banks'])):
        if now['pending'] is not None and was['pending'] is None:
            out.append(dict(event='armed', bank=index, name=now['pending'],
                            score=now['score_at_swap'], games=now['games']))
        if now['opponent'] != was['opponent'] or (was['pending'] is not None and now['pending'] is None):
            retired = next(m for m in ledger.values() if m['name'] == was['opponent'])
            wins, losses, draws = _wld(retired['games'], retired['wins'], retired['draws'])
            out.append(dict(event='loaded', bank=index, name=now['opponent'],
                            epochs_to_align=now['epochs_to_align'],
                            retired=dict(name=was['opponent'], games=retired['games'],
                                         winrate=retired['wins'] / retired['games'] if retired['games'] else None,
                                         score=pfsp.score(wins, losses, draws) if retired['games'] else None)))
    for record in out:
        record.update(learner_steps=learner_steps)
    return out
