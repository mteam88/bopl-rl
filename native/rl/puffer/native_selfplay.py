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


# ---- league: pinned members and prioritised sampling (POLICY_V4_PLAN.md 7e) ----
PFSP_POWER = 2       # weight (1 - winrate)^power: opponents the learner does not beat are drawn most
PFSP_FLOOR = 0.01    # a beaten member is still drawn now and then (forgetting check)


def add_pinned(pool, ledger, pinned):
    """Checkpoints of other runs as permanent pool members.

    `pinned` is [(name, path)]. They sit at the front of the pool (never in
    the five-newest exclusion), are never evicted (see install_sampler) and
    carry ledger steps -1, -2, ... so pool.jsonl / metrics keep one key per
    member. Their Elo starts at the learner's.
    """
    for index, (name, path) in enumerate(pinned):
        if any(entry['path'] == path for entry in pool['pool']):
            raise ValueError(f'pinned opponent {name} is already in the pool')
        pool['pool'].insert(index, dict(path=str(path), elo=pool['primary_elo'], pinned=True, name=name, draws=0))
        ledger[str(path)] = dict(step=-(index + 1), learner_steps=0, games=0, wins=0, name=name, pinned=True)
    pool['pinned'] = [name for name, _ in pinned]


def pfsp_weights(entries, ledger):
    """(1 - p)^PFSP_POWER per entry, p the learner's smoothed lifetime win rate against it.

    The Laplace smoothing gives an unplayed member p = 0.5 (weight 0.25): drawn
    like an even opponent until the ledger says otherwise.
    """
    weights = []
    for entry in entries:
        member = ledger.get(entry['path'], {})
        p = (member.get('wins', 0) + 1.0) / (member.get('games', 0) + 2.0)
        weights.append(max((1.0 - p) ** PFSP_POWER, PFSP_FLOOR))
    return np.array(weights, dtype=np.float64)


def candidates(pool_entries):
    """Pinned members plus the run's own snapshots without the five newest (upstream's rule)."""
    pinned = [entry for entry in pool_entries if entry.get('pinned')]
    own = [entry for entry in pool_entries if not entry.get('pinned')]
    return pinned + (own if len(own) < 6 else own[:-5])


def install_sampler(selfplay, pool, ledger, *, mode):
    """Route pufferlib.selfplay's module-level sampler / eviction through the pool's league rules.

    mode 'index' keeps upstream's (i + 1)^0.5 draw over the candidates; 'pfsp'
    draws by pfsp_weights. Either way pinned members are candidates every draw,
    never evicted, and max_size bounds the run's own snapshots only. Every draw
    is counted in the ledger ('draws'). Idempotent; the trainer calls it once
    per process, after the pool exists (fresh or resumed).
    """
    if mode not in ('index', 'pfsp'):
        raise ValueError(f'unknown opponent sampler {mode!r}')
    upstream_evict = getattr(selfplay, '_bopl_upstream_evict', None) or selfplay.evict
    selfplay._bopl_upstream_evict = upstream_evict

    def sample_opponent(pool_entries, rng):
        chosen = candidates(pool_entries)
        if mode == 'pfsp':
            weights = pfsp_weights(chosen, ledger)
        else:
            weights = np.array([(i + 1) ** 0.5 for i in range(len(chosen))], dtype=np.float64)
        weights /= weights.sum()
        entry = chosen[int(rng.choice(len(chosen), p=weights))]
        # On the entry, not the ledger: a snapshot taken and drawn in the same
        # selfplay.step is registered in the ledger only afterwards.
        entry['draws'] = entry.get('draws', 0) + 1
        return entry

    def evict(pool_entries, max_size):
        pinned = [entry for entry in pool_entries if entry.get('pinned')]
        own = [entry for entry in pool_entries if not entry.get('pinned')]
        return pinned + upstream_evict(own, max_size)

    selfplay.sample_opponent = sample_opponent
    selfplay.evict = evict
    pool['sampler'] = mode


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
    out = dict(schema=1, elo=pool['primary_elo'], size=len(pool['pool']),
               entries=[_member(ledger, entry['path'], entry['elo']) for entry in pool['pool']],
               winrate=sum(bank['hist_score'] for bank in pool['banks'])/games if games else None, banks=banks)
    if pool.get('pinned'):
        # League diagnostic (not a claim): lifetime games / win rate against
        # each pinned member, and their share of the sampler's draws and of
        # all pool-opponent games so far.
        pinned = [entry for entry in pool['pool'] if entry.get('pinned')]
        draws = sum(entry.get('draws', 0) for entry in pool['pool'])
        played = sum(member['games'] for member in ledger.values())
        rows = []
        for entry in pinned:
            member = ledger[entry['path']]
            rows.append(dict(name=entry['name'], elo=entry['elo'], games=member['games'], wins=member['wins'],
                             winrate=member['wins']/member['games'] if member['games'] else None,
                             draws=entry.get('draws', 0)))
        out['sampler'] = pool.get('sampler', 'index')
        out['pinned'] = rows
        out['pinned_draw_share'] = sum(r['draws'] for r in rows)/draws if draws else None
        out['pinned_game_share'] = sum(r['games'] for r in rows)/played if played else None
        out['pinned_banks'] = sum(1 for bank in pool['banks'] if ledger[bank['cur_opp_path']].get('pinned'))
    return out


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
