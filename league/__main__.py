"""The league CLI: seed, add, play, rate, status, candidate, check.

    python league seed
    python league play  --members all --games 24
    python league rate
    python league status
    python league add --name X --kind exploiter --weights policy.bin --run DIR --step N
    python league candidate <run>/checkpoints/policy-N.bin
    python league check

`play` and `candidate` are the verbs that cost CPU; everything else reads the two
committed files. Every game is played under registry.rating_condition (games.py):
side-swapped pairs on fixed seeds, scored off the simulator's own end-of-episode
flags (eval/elo_ladder.play_games), never off a stored `win` field -- the legacy
field recorded a mutual death as a win for whichever side wrote the report.
"""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'eval'), str(ROOT / 'native/tools')]

import numpy as np  # noqa: E402

import registry  # noqa: E402
import games  # noqa: E402
import pfsp  # noqa: E402
import rating  # noqa: E402
import map_catalogue  # noqa: E402

SWEEPS = ROOT / 'native/artifacts/rl/sweeps'
SPECIALISTS = ['dash-bow-platform', 'roll-grenade-smoke', 'teleport-mine-tesla', 'web-mjolnir-freeze',
               'dash-missile-platform', 'grapple-beam-rock', 'duplicator-blink-blackhole', 'drill-throw-magnet']
SEED_STEP = 9253158912
# Reward semantics live in the training binding, not the simulator: f65c33b7
# changed train/bopl_agent.c and bopl_puffer.h only. A run's
# native_binding_sha256 therefore names which semantics it trained under, and is
# the one field that actually discriminates them -- config.json's `reward` dict
# holds the same numbers either side of the change.
BINDINGS = {'c82375d93f4e0478dfebf6b823264072b9e359ac1e9184d2902ecc3b415f4a2c': 'trades-paid'}


def reward_tag(config):
    binding = config.get('native_binding_sha256') or ''
    return dict(tag=BINDINGS.get(binding, f'binding-{binding[:8]}' if binding else 'unknown'),
                binding_sha256=binding or None, shaping=config.get('reward'))


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


# ---- seed --------------------------------------------------------------------

def command_seed(args):
    from evaluation_protocol import SCRIPTED_NAMES, DEFAULT_LOADOUT
    today = date.today().isoformat()
    members = [registry.member('random', 'random', id='random',
                               added=dict(date=today, rule='seed: the anchor, pinned at 0 Elo and never evicted'))]
    for name in SCRIPTED_NAMES:
        members.append(registry.member(name, 'scripted', id=name,
                                       loadouts=dict(mode='fixed', fixed=list(DEFAULT_LOADOUT)),
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
    payload = registry.write(members, {})
    print(f"wrote {registry.LEAGUE.relative_to(ROOT)}: {len(members)} members "
          f"({sum(m['kind']=='specialist' for m in members)} specialist, "
          f"{sum(m['kind']=='scripted' for m in members)} scripted, 1 random)")
    print(f"league_sha256 {payload['league_sha256'][:16]}")
    if not registry.PAYOFF.exists():
        registry.PAYOFF.write_text('')
        print(f"created empty {registry.PAYOFF.relative_to(ROOT)}")


# ---- add ---------------------------------------------------------------------

def command_add(args):
    league = registry.load()
    rows = registry.by_name(league)
    if args.name in rows:
        raise SystemExit(f'{args.name} is already a member; members are immutable')
    # The run's config.json, else the config the checkpoint's own sidecar carries.
    config = (json.loads((Path(args.run_dir) / 'config.json').read_text()) if args.run_dir
              else json.loads(Path(args.weights + '.json').read_text()).get('config') or {})
    member = registry.checkpoint_member(
        args.name, args.kind, Path(args.weights).resolve(), parent=args.parent, target=args.target,
        origin=dict(run=args.run_dir, step=args.step) if args.run_dir else None,
        sim=(config.get('runtime') or {}).get('player_binary_sha256'),
        reward=reward_tag(config) if config else None,
        loadouts=registry.trained_loadouts(config),
        added=dict(date=date.today().isoformat(), rule=args.rule))
    registry.write(league['members'] + [member], league.get('conditions', {}))
    print(f"added {args.name} ({args.kind}) id {member['id'][:16]}")


# ---- play --------------------------------------------------------------------

def command_play(args):
    """Play every informative pair and append the counts to payoff.jsonl.

    A pair continues from the seed pairs it already has under the condition
    (games.matchup), so playing again adds new games rather than replaying old ones.
    """
    league = registry.load()
    print(f"verified {league['verified']['weights_rehashed']} member weights by rehash"
          + (f", missing {league['verified']['missing']}" if league['verified']['missing'] else ''))
    names = select(league, args.members)
    rows = registry.by_name(league)
    key, detail = registry.rating_condition()
    known = league.get('conditions', {})
    if key not in known:
        registry.write(league['members'], dict(known, **{key: detail}))
        print(f'registered new condition {key}')
    counts = registry.read_payoff(condition=key)
    maps = map_catalogue.parse_map_ids(registry.MAPS)
    schedule, skipped = {}, 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ib = rows[a]['id'], rows[b]['id']
            recorded = counts.get((ia, ib) if ia < ib else (ib, ia), [0, 0, 0])
            if args.gate and not pfsp.informative(*recorded):
                skipped += 1
                continue
            tasks = games.matchup(rows[a], rows[b], maps, args.games // 2, args.seed_base, sum(recorded) // 2)
            if tasks:
                schedule[(a, b)] = tasks
    total = sum(len(tasks) for tasks in schedule.values())
    print(f'{key}: {len(names)} members, {len(schedule)} pairs to play, {skipped} skipped as '
          f'uninformative (whole CI outside {pfsp.BAND}); {total} games')
    if not schedule:
        return
    import elo_ladder
    games.configure(Path(args.workspace) / 'workers')
    built = {rows[name]['id']: games.participant(rows[name]) for name in names}
    # Rows are keyed by member id, never by name: payoff.jsonl is append-only and
    # outlives its members, so a reused name would inherit the old agent's record.
    started, played, opponents = time.time(), 0, {}
    with registry.PAYOFF.open('a') as handle:
        for number, ((a, b), tasks) in enumerate(schedule.items(), 1):
            ia, ib = rows[a]['id'], rows[b]['id']
            results = games.play(built, tasks, args.workers, args.seed_base)
            # wins, losses and draws all come from tally(), which counts each
            # game's own outcome. Never `games - wins`: that silently turns every
            # trade and every timeout into a loss for one side.
            record = elo_ladder.tally(results).get((ia, ib), [0, 0, 0])
            row_a, row_b, wins, losses, draws = registry.canonical(ia, ib, *record)
            handle.write(json.dumps(dict(a=row_a, b=row_b, condition=key, wins=wins,
                                         losses=losses, draws=draws)) + '\n')
            handle.flush()
            played += len(results)
            opponents[a] = opponents.get(a, 0) + len(results)
            opponents[b] = opponents.get(b, 0) + len(results)
            if number % 10 == 0 or number == len(schedule):
                rate = played / max(1e-9, time.time() - started)
                print(f'  [{number}/{len(schedule)}] {played} games, {rate:.1f} games/s,'
                      f' {(total-played)/max(rate,1e-9)/60:.0f} min left', flush=True)
    shares = {n: c / max(1, sum(opponents.values())) for n, c in opponents.items()}
    print(f'played {played} games in {(time.time()-started)/60:.1f} min')
    print('realized opponent share: top '
          + ', '.join(f'{n} {s:.3f}' for n, s in sorted(shares.items(), key=lambda kv: -kv[1])[:5]))
    print(f'realized share entropy {pfsp.entropy(shares):.3f} of max {np.log(len(shares)):.3f}')


def command_candidate(args):
    """Rate checkpoints against the league (candidate.rate) and print one JSON line each."""
    import candidate
    for checkpoint in args.checkpoints:
        result = candidate.rate(checkpoint, games=args.games, first_pass_games=args.first_pass_games,
                                band=(args.band_low, args.band_high), round_games=args.round_games,
                                workers=args.workers, seed_base=args.seed_base, source=args.source)
        print(json.dumps(dict(checkpoint=str(checkpoint), rating=round(result['rating'], 1),
                              interval=[round(v, 1) for v in result['interval']], games=result['games'],
                              source=result['source'], condition=result['condition'],
                              kits={kit: round(row['rating'], 1) for kit, row in (result.get('kits') or {}).items()})),
              flush=True)


# ---- rate --------------------------------------------------------------------

def readout(condition=None, members=None, bootstrap=400):
    league = registry.load()
    present = registry.conditions_present()
    if not present:
        raise SystemExit('payoff.jsonl is empty; run `play` first')
    condition = condition or registry.rating_condition()[0]
    counts = registry.read_payoff(condition=condition)
    # Everything below joins on member id; `label` is the only place names re-enter.
    rows = registry.by_id(league)
    names = (registry.ids_for(league, members) if members
             else sorted({n for pair in counts for n in pair}))
    unknown = [n for n in names if n not in rows]
    if unknown:
        raise SystemExit(f'payoff.jsonl references {len(unknown)} participants that are not '
                         f'members of this league: {unknown[:3]}')
    label = {i: rows[i]['name'] for i in names}
    scores, games = rating.matrix(names, counts)
    fit = rating.ratings(names, counts)
    # With no bootstrap budget, fall back to the Wald intervals the fit already
    # produced rather than reporting an interval of zero width.
    boot = (rating.bootstrap_ratings(names, counts, samples=bootstrap) if bootstrap > 0
            else fit['intervals'])
    antisym = rating.advantage(scores)
    mixture = rating.maxent_nash(antisym)
    return dict(league=league, condition=condition, names=names, counts=counts, scores=scores,
                games=games, fit=fit, bootstrap=boot, antisym=antisym, mixture=mixture, label=label,
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
    names, fit, boot, label = data['names'], data['fit'], data['bootstrap'], data['label']
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
        print(f'{label[name]:<28}{fit["ratings"][name]:>8.0f}  [{low:>7.0f},{high:>7.0f}]'
              f'{nash[name]:>+9.3f}{mass[name]:>8.3f}{played:>8}')
    print(f"\nBradley-Terry: anchor {rating.ANCHOR} at 0 Elo, {fit['ci']} intervals shown as bootstrap")
    warning = sparse_warning(cover)
    if warning:
        print('\n' + warning)
    identity = float(data['mixture'] @ data['antisym'] @ data['mixture'])
    print(f'Nash: maxent equilibrium over a_ij = w_ij - 1/2; support '
          f'{int((data["mixture"] > 1e-3).sum())}/{len(names)} members. '
          f'Single-population value n.A.n = {identity:+.2e} (0 by antisymmetry, not a readout).')
    learned = [n for n in names if registry.by_id(data['league'])[n]['kind'] not in ('scripted', 'random')]
    anchors = [n for n in names if n not in learned]
    if learned and anchors:
        cross = data['antisym'][np.ix_([index[n] for n in learned], [index[n] for n in anchors])]
        value = rating.rpp(cross)
        print(f"RPP (learned population vs {len(anchors)} scripted/random anchors): "
              f"{value['value']:+.4f} in score units, i.e. the learned side's equilibrium "
              f"mixture beats the anchors' at {0.5+value['value']:.3f}.")


def run_opponent_shares(directory):
    """The opponent distribution a training run ACTUALLY played, by games.

    rating.saturation asks for the shares that were played, not the weights
    that were intended, and until PFSP landed nothing recorded them. The
    trainer's pool/state.json ledger now carries games/wins/draws per snapshot
    and pool/roster.json the same for the scripted entries; between them that is
    the whole realized distribution. Returns {} when the run has played nothing,
    so the caller reports None rather than a confident zero entropy.
    """
    directory = Path(directory)
    state = json.loads((directory / 'pool' / 'state.json').read_text())
    counts = {Path(path).name: int(row.get('games') or 0)
              for path, row in (state.get('ledger') or {}).items()}
    roster = directory / 'pool' / 'roster.json'
    if roster.is_file():
        for entry in json.loads(roster.read_text()).get('entries') or []:
            counts[entry['name']] = int(entry.get('games') or 0)
    total = sum(counts.values())
    return {name: games / total for name, games in counts.items() if games} if total else {}


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
          f"{cover['games_total']} games, {cover['games_per_pair_min']} games per pair at the thinnest")
    newest = registry.ids_for(data['league'], [args.newest])[0] if args.newest else None
    shares = run_opponent_shares(args.run_dir) if args.run_dir else None
    health = rating.saturation(names, data['counts'], elo=data['fit']['ratings'],
                               mixture=data['mixture'], opponent_shares=shares, newest=newest)
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
        print('  realized opponent distribution: none recorded '
              + ('(that run has played nothing yet)' if args.run_dir
                 else '(pass --run to read one from a training run\'s pool)'))
    else:
        print(f"  realized opponent distribution over {len(shares)} opponents: entropy "
              f"{health['opponent_entropy']:.2f}/{health['opponent_entropy_max']:.2f} nats, "
              f"top share {health['opponent_top_share']:.0%} "
              f"-- this is what was played, not what the pool intended")
    if health.get('ranking_shift'):
        shift = health['ranking_shift']
        print(f"  dropping {data['label'][shift['member']]}: kendall tau {shift['kendall_tau']:.4f}, "
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
    add.add_argument('--run', dest='run_dir')
    add.add_argument('--step', type=int)
    add.add_argument('--parent', help='member the weights were forked from; omit for random init')
    add.add_argument('--target', help='member an exploiter was trained to beat')
    add.add_argument('--rule', default='manual')
    add.set_defaults(run=command_add)

    play = sub.add_parser('play', help='play matchups and append aggregated counts')
    play.add_argument('--members', default='all')
    play.add_argument('--games', type=int, default=24, help='new games per pair this session (even)')
    play.add_argument('--workers', type=int, default=3)
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
    # dest is not `run`: every subparser stores its handler there (set_defaults below).
    status.add_argument('--run', dest='run_dir',
                        help="training run directory whose pool/ supplies the realized "
                             "opponent distribution (otherwise it reads as unrecorded)")
    status.set_defaults(run=command_status)

    rate_one = sub.add_parser('candidate', help='rate checkpoints against the league')
    rate_one.add_argument('checkpoints', nargs='+')
    rate_one.add_argument('--games', type=int, default=240)
    rate_one.add_argument('--first-pass-games', type=int, default=6)
    rate_one.add_argument('--round-games', type=int, default=48)
    rate_one.add_argument('--band-low', type=float, default=.15)
    rate_one.add_argument('--band-high', type=float, default=.85)
    rate_one.add_argument('--workers', type=int, default=4)
    rate_one.add_argument('--seed-base', type=int, default=70001)
    rate_one.add_argument('--source', default='band', choices=('band', 'in-run'))
    rate_one.set_defaults(run=command_candidate)

    sub.add_parser('check', help='prove the estimators on constructed matrices').set_defaults(run=command_check)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == '__main__':
    main()
