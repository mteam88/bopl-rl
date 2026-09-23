"""The league CLI: seed, add, play, rate, status, check.

    python native/rl/league seed
    python native/rl/league play  --members specialist,random,scripted:hard --games 48
    python native/rl/league rate
    python native/rl/league status
    python native/rl/league add --name X --kind exploiter --weights policy.bin --run DIR --step N
    python native/rl/league check

`play` is the only verb that costs CPU. Everything else reads the two committed
files. Games are always side-swapped pairs on fixed seeds and are scored off the
simulator's own end-of-episode flags (research/elo_ladder.play_games), never off
a stored `win` field -- the legacy field recorded a mutual death as a win for
whichever side wrote the report.
"""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]

import numpy as np  # noqa: E402

import registry  # noqa: E402
import pfsp  # noqa: E402
import rating  # noqa: E402

SIM = ROOT / 'native/target/release/libbopl_sim.so'
SWEEPS = ROOT / 'native/artifacts/rl/sweeps'
LINEUPS8 = ROOT / 'native/data/lineups-8.txt'
SPECIALISTS = ['dash-bow-platform', 'roll-grenade-smoke', 'teleport-mine-tesla', 'web-mjolnir-freeze',
               'dash-missile-platform', 'grapple-beam-rock', 'duplicator-blink-blackhole', 'drill-throw-magnet']
SEED_STEP = 9253158912
# Reward semantics live in the training binding, not the simulator: f65c33b7
# changed native/rl/puffer/bopl_agent.c and bopl_puffer.h only. A run's
# native_binding_sha256 therefore names which semantics it trained under, and is
# the one field that actually discriminates them -- config.json's `reward` dict
# holds the same numbers either side of the change.
BINDINGS = {'c82375d93f4e0478dfebf6b823264072b9e359ac1e9184d2902ecc3b415f4a2c': 'trades-paid'}


def reward_tag(config):
    binding = config.get('native_binding_sha256') or ''
    return dict(tag=BINDINGS.get(binding, f'binding-{binding[:8]}' if binding else 'unknown'),
                binding_sha256=binding or None, shaping=config.get('reward'))


def configure_environment(workspace):
    """Point the research C environment at the tree's Rust sim, as train.py does."""
    os.environ.update(BOPL_RL_BACKEND='rust', BOPL_RL_INPROC_SPARE='0',
                      BOPL_RL_WORLDS_PER_RUNTIME=str(1 << 20), OMP_NUM_THREADS='1')
    from environment import configure
    configure(Path(workspace) / 'workers', SIM)


def seed_loadouts():
    from evaluation_protocol import resolve_loadouts
    return resolve_loadouts(lineup_file=str(LINEUPS8), mirror_share=1.0)


# ---- selection ---------------------------------------------------------------

def select(league, tokens):
    """Member names from comma-separated tokens: a name, a kind, `scripted:LEVEL`, or `all`."""
    members = league['members']
    chosen = []
    for token in (t.strip() for t in tokens.split(',') if t.strip()):
        if token == 'all':
            picked = [m['name'] for m in members]
        elif token in registry.KINDS:
            picked = [m['name'] for m in members if m['kind'] == token]
        elif token.startswith('scripted:'):
            level = token.split(':', 1)[1]
            picked = [m['name'] for m in members if m['kind'] == 'scripted' and m['name'].endswith('-' + level)]
        else:
            picked = [m['name'] for m in members if m['name'] == token]
            if not picked:
                raise SystemExit(f'no member, kind or level matches {token!r}')
        chosen += [name for name in picked if name not in chosen]
    return chosen


def participants(league, names, library=None):
    """elo_ladder participants for the chosen members."""
    import elo_ladder
    rows = registry.by_name(league)
    built = {}
    for name in names:
        row = rows[name]
        if row['kind'] == 'human':
            raise SystemExit(
                f'{name} is a human member: their games are recorded, not played. '
                f'Append their counts to payoff.jsonl directly and exclude them from `play`.')
        if row['kind'] == 'random':
            built[name] = elo_ladder.Random()
        elif row['kind'] == 'scripted':
            built[name] = elo_ladder.Scripted(name)
        else:
            built[name] = elo_ladder.Model(name, ROOT / row['path'], row['kind'], library=library)
    return built


# ---- seed --------------------------------------------------------------------

def command_seed(args):
    from evaluation_protocol import SCRIPTED_NAMES
    today = date.today().isoformat()
    members = [registry.member('random', 'random', id='random',
                               added=dict(date=today, rule='seed: the anchor, pinned at 0 Elo and never evicted'))]
    for name in SCRIPTED_NAMES:
        members.append(registry.member(name, 'scripted', id=name,
                                       added=dict(date=today, rule='seed: hand-coded roster (bopl_bots.h)')))
    for kit in SPECIALISTS:
        run = SWEEPS / f'{kit}-long/000'
        config = json.loads((run / 'config.json').read_text())
        weights = run / f'checkpoints/policy-{SEED_STEP}.bin'
        members.append(registry.checkpoint_member(
            kit, 'specialist', weights,
            parent=None,   # native.load_model_path is null: these trained from random init
            origin=dict(run=str(run.relative_to(ROOT)), step=SEED_STEP),
            sim=(config.get('runtime') or {}).get('player_binary_sha256'),
            reward=reward_tag(config),
            loadouts=dict(mode='fixed', fixed=config['loadouts']['fixed']),
            added=dict(date=today, rule='seed: strongest known play on its kit, frozen as an exploiter')))
    _, detail = registry.condition(registry.sha256_file(SIM), seed_loadouts(), [0, 1, 2])
    payload = registry.write(members, {detail['key']: detail})
    print(f"wrote {registry.LEAGUE.relative_to(ROOT)}: {len(members)} members "
          f"({sum(m['kind']=='specialist' for m in members)} specialist, "
          f"{sum(m['kind']=='scripted' for m in members)} scripted, 1 random)")
    print(f"league_sha256 {payload['league_sha256'][:16]}  condition {detail['key']}")
    if not registry.PAYOFF.exists():
        registry.PAYOFF.write_text('')
        print(f"created empty {registry.PAYOFF.relative_to(ROOT)}")


# ---- add ---------------------------------------------------------------------

def command_add(args):
    league = registry.load()
    rows = registry.by_name(league)
    if args.name in rows:
        raise SystemExit(f'{args.name} is already a member; members are immutable')
    config = json.loads((Path(args.run) / 'config.json').read_text()) if args.run else {}
    member = registry.checkpoint_member(
        args.name, args.kind, Path(args.weights).resolve(), parent=args.parent, target=args.target,
        origin=dict(run=args.run, step=args.step) if args.run else None,
        sim=(config.get('runtime') or {}).get('player_binary_sha256'),
        reward=reward_tag(config) if config else None,
        loadouts=(dict(mode=config['loadouts']['mode'], fixed=config['loadouts'].get('fixed'),
                       set_file=config['loadouts'].get('set_file')) if config.get('loadouts') else None),
        added=dict(date=date.today().isoformat(), rule=args.rule))
    registry.write(league['members'] + [member], league.get('conditions', {}))
    print(f"added {args.name} ({args.kind}) id {member['id'][:16]}")


# ---- play --------------------------------------------------------------------

def command_play(args):
    import elo_ladder
    league = registry.load()
    print(f"verified {league['verified']['weights_rehashed']} member weights by rehash"
          + (f", missing {league['verified']['missing']}" if league['verified']['missing'] else ''))
    names = select(league, args.members)
    loadouts = seed_loadouts()
    key, detail = registry.condition(registry.sha256_file(SIM), loadouts, args.maps)
    known = league.get('conditions', {})
    if key in known and known[key]['loadout_sha256'] != detail['loadout_sha256']:
        raise SystemExit(f'condition {key} already names a different loadout set; refusing to pool them')
    if key not in known:
        registry.write(league['members'], dict(known, **{key: detail}))
        print(f'registered new condition {key}')

    counts = registry.read_payoff(condition=key)
    pairs, skipped = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            value = counts.get((a, b) if a < b else (b, a), [0, 0, 0])
            wins, losses, draws = value if a < b else [value[1], value[0], value[2]]
            if args.gate and not pfsp.informative(wins, losses, draws):
                skipped.append(((a, b), pfsp.score(wins, losses, draws), wins + losses + draws))
            else:
                pairs.append((a, b))
    print(f'{len(names)} members, {len(pairs)} pairs to play, {len(skipped)} skipped as uninformative'
          f' (whole CI outside {pfsp.BAND})')
    if not pairs:
        return
    configure_environment(args.workspace)
    built = participants(league, names)
    lineups = list(range(len(loadouts['lineups'])))
    total = len(pairs) * len(args.maps) * len(lineups) * args.games_per_cell
    print(f'{total} games = {len(pairs)} pairs x {len(args.maps)} maps x {len(lineups)} lineups'
          f' x {args.games_per_cell} (side-swapped); est {total*0.2/60:.0f} min at 0.2 s/game')

    started, played, opponents, cells = time.time(), 0, {}, set()
    with registry.PAYOFF.open('a') as handle:
        for number, (a, b) in enumerate(pairs, 1):
            tasks = elo_ladder.pair_tasks(a, b, list(args.maps), args.games_per_cell,
                                          args.seed_base, lineups=lineups)
            games = elo_ladder.play_games(built, tasks, args.episode_ticks, args.workers,
                                          args.seed_base, loadouts=loadouts)
            # wins, losses and draws all come from tally(), which counts each
            # game's own outcome. Never `games - wins`: that silently turns every
            # trade and every timeout into a loss for one side.
            record = elo_ladder.tally(games).get((a, b), [0, 0, 0])
            # Prove the session exercised the condition it claims: record which
            # (map, loadout) cells the games actually landed on, not which ones
            # the schedule intended.
            cells |= {(g['map'], tuple(g.get('lineup_ids', ()))) for g in games}
            row_a, row_b, wins, losses, draws = registry.canonical(a, b, *record)
            handle.write(json.dumps(dict(a=row_a, b=row_b, condition=key, wins=wins,
                                         losses=losses, draws=draws)) + '\n')
            handle.flush()
            played += len(games)
            opponents[a] = opponents.get(a, 0) + len(games)
            opponents[b] = opponents.get(b, 0) + len(games)
            if number % 10 == 0 or number == len(pairs):
                rate = played / max(1e-9, time.time() - started)
                print(f'  [{number}/{len(pairs)}] {played} games, {rate:.1f} games/s,'
                      f' {(total-played)/max(rate,1e-9)/60:.0f} min left', flush=True)
    shares = {n: c / max(1, sum(opponents.values())) for n, c in opponents.items()}
    print(f'played {played} games in {(time.time()-started)/60:.1f} min')
    print('realized opponent share: top '
          + ', '.join(f'{n} {s:.3f}' for n, s in sorted(shares.items(), key=lambda kv: -kv[1])[:5]))
    print(f'realized share entropy {pfsp.entropy(shares):.3f} of max {np.log(len(shares)):.3f}')
    maps_hit = sorted({cell[0] for cell in cells})
    kits_hit = sorted({cell[1] for cell in cells})
    expected = len(args.maps) * len(loadouts['lineups'])
    print(f'exercised {len(cells)}/{expected} (map, loadout) cells: maps {maps_hit}, '
          f'{len(kits_hit)} distinct loadouts of {len(loadouts["lineups"])}'
          + ('' if len(cells) == expected else '  <-- INCOMPLETE, some cells never played'))


# ---- rate --------------------------------------------------------------------

def readout(condition=None, members=None, bootstrap=400):
    league = registry.load()
    present = registry.conditions_present()
    if not present:
        raise SystemExit('payoff.jsonl is empty; run `play` first')
    condition = condition or max(present, key=present.get)
    counts = registry.read_payoff(condition=condition)
    names = members or sorted({n for pair in counts for n in pair})
    scores, games = rating.matrix(names, counts)
    fit = rating.ratings(names, counts)
    # With no bootstrap budget, fall back to the Wald intervals the fit already
    # produced rather than reporting an interval of zero width.
    boot = (rating.bootstrap_ratings(names, counts, samples=bootstrap) if bootstrap > 0
            else fit['intervals'])
    antisym = rating.advantage(scores)
    mixture = rating.maxent_nash(antisym)
    return dict(league=league, condition=condition, names=names, counts=counts, scores=scores,
                games=games, fit=fit, bootstrap=boot, antisym=antisym, mixture=mixture,
                nash=rating.nash_scores(antisym, mixture),
                coverage=rating.coverage(names, games))


def sparse_warning(cover):
    """The caveat that has to travel with every Nash number on a partial matrix.

    Bradley-Terry only reads pairs that exist, so a sparse matrix costs it
    precision and nothing else. Nash and RPP read the whole matrix, and
    rating.advantage fills what was never played with a neutral 0, so a mostly
    unplayed league has a near-uniform equilibrium and every Nash score collapses
    toward 0.000 -- or, just as misleadingly, onto a single arbitrary member.
    That is an artefact of the holes, not a finding about the members.
    """
    if cover['fraction'] >= .9:
        return None
    return (f"  WARNING: {1 - cover['fraction']:.0%} of the matrix was never played and is filled\n"
            f"  with a neutral 0. Bradley-Terry is unaffected; the Nash mixture, Nash scores and\n"
            f"  RPP below are dominated by that fill and should not be read as results until the\n"
            f"  missing {cover['pairs_missing']} pairs are played.")


def command_rate(args):
    data = readout(args.condition, bootstrap=args.bootstrap)
    names, fit, boot = data['names'], data['fit'], data['bootstrap']
    order = sorted(names, key=lambda n: -fit['ratings'][n])
    print(f"condition {data['condition']}")
    cover = data['coverage']
    print(f"coverage {cover['pairs_played']}/{cover['pairs_possible']} pairs "
          f"({cover['fraction']:.0%}), {cover['games_total']} games, "
          f"median {cover['games_per_pair_median']:.0f} per played pair")
    print(f"\n{'member':<28}{'Elo':>8}{'  bootstrap 95%':>20}{'   Nash':>9}{'  mass':>8}{'  games':>8}")
    mass = dict(zip(names, data['mixture']))
    nash = dict(zip(names, data['nash']))
    index = {n: i for i, n in enumerate(names)}
    for name in order:
        low, high = boot[name]
        played = int(data['games'][index[name]].sum())
        print(f'{name:<28}{fit["ratings"][name]:>8.0f}  [{low:>7.0f},{high:>7.0f}]'
              f'{nash[name]:>+9.3f}{mass[name]:>8.3f}{played:>8}')
    print(f"\nBradley-Terry: anchor {rating.ANCHOR} at 0 Elo, {fit['ci']} intervals shown as bootstrap")
    warning = sparse_warning(cover)
    if warning:
        print('\n' + warning)
    identity = float(data['mixture'] @ data['antisym'] @ data['mixture'])
    print(f'Nash: maxent equilibrium over a_ij = w_ij - 1/2; support '
          f'{int((data["mixture"] > 1e-3).sum())}/{len(names)} members. '
          f'Single-population value n.A.n = {identity:+.2e} (0 by antisymmetry, not a readout).')
    learned = [n for n in names if registry.by_name(data['league'])[n]['kind'] not in ('scripted', 'random')]
    anchors = [n for n in names if n not in learned]
    if learned and anchors:
        cross = data['antisym'][np.ix_([index[n] for n in learned], [index[n] for n in anchors])]
        value = rating.rpp(cross)
        print(f"RPP (learned population vs {len(anchors)} scripted/random anchors): "
              f"{value['value']:+.4f} in score units, i.e. the learned side's equilibrium "
              f"mixture beats the anchors' at {0.5+value['value']:.3f}.")


def command_status(args):
    data = readout(args.condition, bootstrap=0 if args.fast else 200)
    league, names = data['league'], data['names']
    kinds = {}
    for row in league['members']:
        kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
    print(f"league {registry.LEAGUE.relative_to(ROOT)}  sha {league['league_sha256'][:16]}")
    print(f"  {len(league['members'])} members: " + ', '.join(f'{v} {k}' for k, v in sorted(kinds.items())))
    print(f"  verified: {league['verified']['weights_rehashed']} weights rehashed, "
          f"{league['verified']['checked']} members checked, missing {league['verified']['missing'] or 'none'}")
    print(f"  conditions in payoff: " + ', '.join(f'{k} ({v} rows)' for k, v in registry.conditions_present().items()))
    cover = data['coverage']
    print(f"\ncoverage under {data['condition']}")
    print(f"  {cover['pairs_played']}/{cover['pairs_possible']} pairs ({cover['fraction']:.0%}), "
          f"{cover['games_total']} games, {cover['games_per_pair_min']} min per pair")
    health = rating.saturation(names, data['counts'], elo=data['fit']['ratings'],
                               mixture=data['mixture'], newest=args.newest)
    print('\nsaturation')
    print(f"  Elo spread {health['elo_spread']:.0f} (range {health['elo_range']:.0f}) "
          f"-- collapsing toward 0 means members stopped differing")
    print(f"  pair scores: sd {health['pair_score_spread']:.3f} around 0.5, "
          f"{health['pairs_near_even']:.0%} of {health['pairs_scored']} pairs within 0.45-0.55")
    print(f"  Nash support {health['nash_support']}/{len(names)} members, entropy "
          f"{health['nash_entropy']:.2f}/{health['nash_entropy_max']:.2f} nats")
    warning = sparse_warning(cover)
    if warning:
        print(warning)
    if health.get('opponent_entropy') is None:
        print('  realized opponent distribution: none recorded (no learner has trained against this league yet)')
    if health.get('ranking_shift'):
        shift = health['ranking_shift']
        print(f"  dropping {shift['member']}: kendall tau {shift['kendall_tau']:.4f}, "
              f"ranking {'changed' if shift['changed_ranking'] else 'UNCHANGED (member added no information)'}, "
              f"max Elo move {shift['max_elo_move']:.0f}")


def command_check(args):
    import check
    raise SystemExit(check.run())


def main(argv=None):
    parser = argparse.ArgumentParser(prog='league', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='verb', required=True)

    sub.add_parser('seed', help='write league.json with the specialists, scripted roster and anchor').set_defaults(run=command_seed)

    add = sub.add_parser('add', help='append one immutable member')
    add.add_argument('--name', required=True)
    add.add_argument('--kind', required=True, choices=registry.KINDS)
    add.add_argument('--weights', required=True)
    add.add_argument('--run')
    add.add_argument('--step', type=int)
    add.add_argument('--parent', help='member the weights were forked from; omit for random init')
    add.add_argument('--target', help='member an exploiter was trained to beat')
    add.add_argument('--rule', default='manual')
    add.set_defaults(run=command_add)

    play = sub.add_parser('play', help='play matchups and append aggregated counts')
    play.add_argument('--members', default='all')
    play.add_argument('--maps', type=lambda s: [int(x) for x in s.split(',')], default=[0, 1, 2])
    play.add_argument('--games-per-cell', type=int, default=2,
                      help='side-swapped games per (map, lineup) cell; must be even')
    play.add_argument('--workers', type=int, default=3)
    play.add_argument('--episode-ticks', type=int, default=7200)
    play.add_argument('--seed-base', type=int, default=700001)
    play.add_argument('--workspace', default='/tmp/league-play')
    play.add_argument('--no-gate', dest='gate', action='store_false',
                      help='play every pair, including ones already outside 15-85%%')
    play.set_defaults(run=command_play, gate=True)

    rate = sub.add_parser('rate', help='Bradley-Terry ratings, Nash averaging and RPP')
    rate.add_argument('--condition')
    rate.add_argument('--bootstrap', type=int, default=400)
    rate.set_defaults(run=command_rate)

    status = sub.add_parser('status', help='members, matrix coverage and the saturation readout')
    status.add_argument('--condition')
    status.add_argument('--newest', help='member to leave out for the ranking-shift test')
    status.add_argument('--fast', action='store_true')
    status.set_defaults(run=command_status)

    sub.add_parser('check', help='prove the estimators on constructed matrices').set_defaults(run=command_check)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == '__main__':
    main()
