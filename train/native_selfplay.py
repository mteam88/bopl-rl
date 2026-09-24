"""PufferLib's selfplay pool (pufferlib.selfplay) on the Bopl trainer's layout.

pufferlib.selfplay.setup assumes every environment has the same number of
agents; Bopl's self-play games take two rows and historical games 1+banks rows
(one per frozen bank, one active). The state built here has exactly the keys
selfplay.step reads, and the trainer calls selfplay.step every update for the
Elo ratings, opponent sampling, snapshot cadence, eviction and bank swaps.

Swaps replace a bank's weights in place while its matches run (the alignment
gate only waits until every environment of the bank has started a match since
the previous swap). The environment therefore drops a match whose opponent
changed mid-match from the pool's per-bank statistics (see bopl_puffer.h).
"""
import numpy as np


def state(pool_dir, bootstrap, banks, historical_per_bank, global_step, *, max_size, swap_winrate,
          min_games, snapshot_interval, opp_timeout_steps, elo_init, elo_k, seed):
    if banks < 1 or historical_per_bank < 1 or max_size < 1 or min_games < 1:
        raise ValueError('The pool needs at least one bank, historical game, entry and game per swap')
    if not 0 < swap_winrate <= 1:
        raise ValueError('swap_winrate must be within (0, 1]')
    bank = dict(cur_opp_path=bootstrap, cur_opp_elo=elo_init, hist_score=0.0, hist_n=0.0,
                pending_opp_path=None, pending_opp_elo=None, epoch_armed=0, opp_started_step=global_step,
                num_hist_envs=historical_per_bank, last_winrate_at_swap=0.0, last_epochs_to_align=0)
    return dict(pool_dir=str(pool_dir), pool=[dict(path=bootstrap, elo=elo_init)],
                rng=np.random.default_rng(seed), max_size=max_size, min_games=min_games,
                swap_winrate=swap_winrate, snapshot_interval=snapshot_interval,
                opp_timeout_steps=opp_timeout_steps, num_banks=banks,
                banks=[dict(bank) for _ in range(banks)], primary_elo=elo_init, elo_k=elo_k,
                last_snapshot_step=global_step)


def members(bootstrap):
    """Ledger of every snapshot the pool ever held: age and lifetime games against the learner."""
    return {bootstrap: dict(step=0, learner_steps=0, games=0, wins=0)}


def credit(pool, flat, ledger):
    """Attribute the window's finished games to each bank's current opponent.

    Call before selfplay.step: it rates the same games, then may swap the bank.
    """
    n = float(flat.get('env/n', 0.0))
    for index, bank in enumerate(pool['banks']):
        games = round(float(flat.get(f'env/hist_n_bank_{index}', 0.0)) * n)
        if games:
            member = ledger[bank['cur_opp_path']]
            member['games'] += games
            member['wins'] += round(float(flat.get(f'env/hist_score_bank_{index}', 0.0)) * n)


def step(selfplay_step, trainer, backend, pool, flat, epoch):
    """pufferlib.selfplay.step, then collapse the duplicate entries it can create.

    When several banks pass the swap gate in the same update (they start
    synchronised on the initial checkpoint), upstream saves and appends the same
    {global_step}.bin once per bank: the copies are rated separately (only the
    first receives the Elo update), sampled with multiplied weight, and count
    towards the winrate-snapshot cap. Keep the first occurrence of each path.
    With max_size >= 10 (winrate snapshots stop at ten entries) the duplicates
    never reach eviction, so the pool is exactly what a single append would give;
    a smaller max_size can still evict one extra old entry in the colliding update.
    """
    selfplay_step(trainer, backend, pool, flat, epoch)
    seen, unique = set(), []
    for entry in pool['pool']:
        if entry['path'] not in seen:
            seen.add(entry['path'])
            unique.append(entry)
    pool['pool'] = unique


def register(pool, ledger, step, learner_steps):
    """Record when snapshots selfplay.step just took were made."""
    for entry in pool['pool']:
        ledger.setdefault(entry['path'], dict(step=step, learner_steps=learner_steps, games=0, wins=0))


def _member(ledger, path, elo):
    member = ledger[path]
    return dict(step=member['step'], learner_steps=member['learner_steps'], elo=elo, games=member['games'],
                winrate=member['wins']/member['games'] if member['games'] else None)


def summary(pool, ledger):
    """The pool's rating/swap state for metrics.jsonl."""
    banks = []
    for bank in pool['banks']:
        games = round(bank['hist_n'])
        banks.append(dict(opponent_step=ledger[bank['cur_opp_path']]['step'], opponent_elo=bank['cur_opp_elo'],
                          games=games, winrate=bank['hist_score']/bank['hist_n'] if games else None,
                          pending_step=ledger[bank['pending_opp_path']]['step'] if bank['pending_opp_path'] else None,
                          winrate_at_swap=bank['last_winrate_at_swap'], epochs_to_align=bank['last_epochs_to_align']))
    games = sum(bank['hist_n'] for bank in pool['banks'])
    return dict(schema=1, elo=pool['primary_elo'], size=len(pool['pool']),
                entries=[_member(ledger, entry['path'], entry['elo']) for entry in pool['pool']],
                winrate=sum(bank['hist_score'] for bank in pool['banks'])/games if games else None, banks=banks)


def events(before, after, ledger, learner_steps):
    """pool.jsonl records for what selfplay.step changed between two summaries."""
    out = []
    old = {entry['step']: entry for entry in before['entries']}
    new = {entry['step']: entry for entry in after['entries']}
    for step in sorted(new.keys() - old.keys()):
        out.append(dict(event='snapshot', step=step))
    for step in sorted(old.keys() - new.keys()):
        out.append(dict(event='evicted', **old[step]))
    for index, (was, now) in enumerate(zip(before['banks'], after['banks'])):
        if now['pending_step'] is not None and was['pending_step'] is None:
            out.append(dict(event='armed', bank=index, step=now['pending_step'],
                            winrate=now['winrate_at_swap'], games=now['games']))
        if now['opponent_step'] != was['opponent_step'] or (was['pending_step'] is not None and now['pending_step'] is None):
            retired = next(m for m in ledger.values() if m['step'] == was['opponent_step'])
            out.append(dict(event='loaded', bank=index, step=now['opponent_step'], elo=now['opponent_elo'],
                            epochs_to_align=now['epochs_to_align'],
                            retired=dict(step=was['opponent_step'], elo=was['opponent_elo'], games=retired['games'],
                                         winrate=retired['wins']/retired['games'] if retired['games'] else None)))
    for record in out:
        record.update(learner_steps=learner_steps, elo=record.get('elo', after['elo']))
    return out
