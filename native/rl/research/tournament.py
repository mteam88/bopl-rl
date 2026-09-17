"""Resumable direct round robin for the top of the joint ladder.

The tournament is emitted as a pool-shaped JSON file, so joint_ladder ingests
its games without a special source or a second rating implementation.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent / 'sweep'), str(ROOT / 'native/tools')]

import elo_ladder
import joint_ladder
import reference_pool
from evaluation_protocol import checkpoint_identity

BASE_ROOT = Path('/home/mte/Documents/Coding/bopl-re/ripped/ExportedProject/native/artifacts/rl')
POOL_V7 = BASE_ROOT / 'sweeps/anchors/pool-v7/pool.json'
POLICY_LIBRARY = Path('/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/policy-v4/native/artifacts/rl/libbopl_policy_eval.so')
RESEARCH_LIBRARY = POLICY_LIBRARY.with_name('libbopl_research.so')
SEED_BASE = 90001


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_ladder(root, extra_pool=None):
    runs = joint_ladder.discover_runs(root)
    pools, issues = joint_ladder.discover_pools(root, extra=[extra_pool] if extra_pool else None)
    fit = joint_ladder.build(root, runs, pools, reference_pool.DEFAULT_HUMAN)
    fit['issues'] = [*issues, *fit['issues']]
    return runs, fit


def checkpoint_path(row, runs):
    run = runs.get(row.get('run'))
    if not run or row.get('steps') is None:
        return None
    return Path(run['path']) / 'checkpoints' / f"policy-{int(row['steps']):09d}.bin"


def unique_name(wanted, used):
    name = wanted
    suffix = 2
    while name in used:
        name = f'{wanted} #{suffix}'
        suffix += 1
    used.add(name)
    return name


def select_roster(top, pool, runs, before, base_out):
    used_names = set()
    seen_sha = set()
    roster = []
    skipped = 0
    selected = 0
    by_id = {row['id']: row for row in before['rows']}
    for row in before['rows']:
        if row.get('kind') != 'checkpoint' or selected >= top:
            continue
        path = checkpoint_path(row, runs)
        if not path or not path.is_file():
            skipped += 1
            continue
        sha = sha256_file(path)
        if sha != row.get('sha256'):
            skipped += 1
            continue
        if sha in seen_sha:
            continue
        name = unique_name(row['name'], used_names)
        identity = checkpoint_identity(path)
        roster.append(dict(name=name, kind='checkpoint', sha256=sha,
                           metadata_sha256=identity['metadata_sha256'], source=path,
                           path=os.path.relpath(path, base_out), before=row))
        seen_sha.add(sha)
        selected += 1
    if selected < top:
        raise RuntimeError(f'only {selected} surviving checkpoint members found for --top {top} ({skipped} missing)')

    pool_dir = Path(pool['path']).parent
    for member in pool['members']:
        sha = member.get('sha256')
        if sha and sha in seen_sha:
            continue
        name = unique_name(member['name'], used_names)
        item = dict(name=name, kind=member['kind'], sha256=sha,
                    before=by_id.get(joint_ladder.member_id(member)))
        if member['kind'] == 'checkpoint':
            path = (pool_dir / member['path']).resolve()
            identity = checkpoint_identity(path)
            item.update(source=path, path=os.path.relpath(path, base_out),
                        metadata_sha256=identity['metadata_sha256'])
            seen_sha.add(sha)
        roster.append(item)
    return roster, skipped


def writer_ablation(member):
    if member['kind'] != 'checkpoint':
        return None
    sidecar = Path(str(member['source']) + '.json')
    meta = json.loads(sidecar.read_text())
    if meta.get('architecture') != 'bopl-v4-mingru':
        return None
    return int((meta.get('v4_shape') or {}).get('ablate', 0))


def writer_groups(roster):
    groups = {}
    for member in roster:
        value = writer_ablation(member)
        if value is not None:
            groups.setdefault(value, []).append(member['name'])
    return groups


def playable(roster, library):
    result = {}
    for member in roster:
        if member['kind'] == 'random':
            result[member['name']] = elo_ladder.Random()
        elif member['kind'] == 'scripted':
            result[member['name']] = elo_ladder.Scripted(member['name'])
        else:
            result[member['name']] = elo_ladder.Model(member['name'], member['source'], 'checkpoint', library=library)
    return result


def load_games(path):
    rows = []
    if path.is_file():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
    return rows


def task_key(row):
    return row['a'], row['b'], int(row['map']), int(row['seed']), int(row['side'])


def plan_tasks(names, maps, games_per_pair, existing):
    done = {task_key(row) for row in existing}
    tasks = []
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            for task in elo_ladder.pair_tasks(a, b, maps, games_per_pair // len(maps), SEED_BASE):
                if task_key(task) not in done:
                    tasks.append(task)
    return tasks


def game_row(game, members):
    a, b = members[game['a']], members[game['b']]
    return dict(a=game['a'], a_sha=a.get('sha256'), a_name=a['name'],
                b=game['b'], b_sha=b.get('sha256'), b_name=b['name'],
                map=int(game['map']), seed=int(game['seed']), side=int(game['side']),
                outcome=game['outcome'], ticks=int(game['ticks']))


def local_fit(roster, games):
    names = [member['name'] for member in roster]
    records = elo_ladder.tally(games)
    fitted = elo_ladder.bradley_terry_fit(names, records, 'random')
    counts = reference_pool.tally_counts(names, records)
    return records, fitted, counts


def public_member(member, fitted, counts):
    result = {key: member[key] for key in ('name', 'kind', 'sha256', 'metadata_sha256', 'path') if member.get(key) is not None}
    wins, losses, draws = counts[member['name']]
    result.update(rating=round(fitted['ratings'][member['name']], 3),
                  interval=fitted['intervals'][member['name']], games=wins + losses + draws,
                  wins=wins, losses=losses, draws=draws)
    return result


def write_pool(out, roster, games, pool_v7, game, games_per_pair, skipped_pairs, elapsed):
    records, fitted, counts = local_fit(roster, games)
    members = [public_member(member, fitted, counts) for member in roster]
    result = dict(
        schema=reference_pool.SCHEMA, created=time.time(),
        note='Direct top-ladder round robin; side-swapped fixed-seed games on pool-v7 maps. '
             f'{skipped_pairs} cross-family v4 pairs are temporarily skipped because the current '
             'observation writer has process-global BOPL4_ABLATE; each v4 family plays every '
             'writer-agnostic bridge, and bridge-only pairs are played once. Local Bradley-Terry '
             'fit with random anchored at 0; Wald 95% intervals.',
        game=dict(path=str(game), sha256=sha256_file(game)), maps=list(pool_v7['maps']),
        games_per_pair=games_per_pair, seed_base=SEED_BASE,
        episode_ticks=int(pool_v7['episode_ticks']),
        observation_delay_ticks=int(pool_v7.get('observation_delay_ticks', 9)),
        games=len(games), elapsed=elapsed,
        fit='Bradley-Terry MLE, timeouts as draws, one virtual draw per pair, random anchored at 0 Elo; Wald 95% CI',
        members=members,
        matches=[dict(a=a, b=b, wins_a=wa, wins_b=wb, draws=draws)
                 for (a, b), (wa, wb, draws) in sorted(records.items())],
    )
    result['pool_sha256'] = reference_pool.pool_hash(members)
    reference_pool.atomic_json(out / 'tournament.json', result)
    return result


def comparison(roster, before, after, tournament):
    before_rows = {row['id']: row for row in before['rows']}
    after_rows = {row['id']: row for row in after['rows']}
    tournament_rows = {joint_ladder.member_id(row): row for row in tournament['members']}
    rows = []
    for member in roster:
        pid = member.get('sha256') or member['name']
        old, new, local = before_rows.get(pid), after_rows.get(pid), tournament_rows.get(pid)
        if not old or not new or not local:
            continue
        rows.append(dict(member=member['name'], id=pid, rank_before=old['rank'], rank_after=new['rank'],
                         rating_before=old['rating'], rating_tournament=local['rating'], rating_after=new['rating'],
                         ci_before=old['interval'][1] - old['interval'][0],
                         ci_after=new['interval'][1] - new['interval'][0],
                         tournament_interval=local['interval']))
    rows.sort(key=lambda row: row['rank_after'])
    return rows


def print_roster(roster, skipped):
    print(f'roster={len(roster)} skipped_missing={skipped}', flush=True)
    print('rank  rating  ci_width  member', flush=True)
    for member in roster:
        row = member.get('before')
        print(f"{row['rank'] if row else '-':>4}  {row['rating'] if row else 0:7.1f}  "
              f"{(row['interval'][1]-row['interval'][0]) if row else 0:8.1f}  {member['name']}", flush=True)


def play_family(roster, family, pool_v7, args, game):
    os.environ['BOPL_RL_BACKEND'] = 'rust'
    os.environ['BOPL4_ABLATE'] = str(family)
    os.environ['BOPL_RESEARCH_LIBRARY'] = str(args.research_library.resolve())
    reference_pool.configure_games(args.out, game)
    subset = [member for member in roster if writer_ablation(member) in (None, family)]
    by_name = playable(subset, args.library.resolve())
    members = {member['name']: member for member in subset}
    games_path = args.out / 'games.jsonl'
    existing = load_games(games_path)
    tasks = plan_tasks(list(by_name), pool_v7['maps'], args.games_per_pair, existing)
    print(f'family={family} members={len(subset)} games_existing={len(existing)} games_missing={len(tasks)}', flush=True)
    lock = threading.Lock()
    with games_path.open('a', buffering=1) as handle:
        def persist(raw):
            row = game_row(raw, members)
            with lock:
                handle.write(json.dumps(row, separators=(',', ':')) + '\n')
                handle.flush()
        elo_ladder.play_games(by_name, tasks, int(pool_v7['episode_ticks']), args.workers,
                              SEED_BASE, delay=int(pool_v7.get('observation_delay_ticks', 9)),
                              on_result=persist)


def schedule_counts(roster, games_per_pair):
    allowed = skipped = 0
    for index, a in enumerate(roster):
        a_writer = writer_ablation(a)
        for b in roster[index + 1:]:
            b_writer = writer_ablation(b)
            if a_writer is not None and b_writer is not None and a_writer != b_writer:
                skipped += 1
            else:
                allowed += 1
    return allowed, skipped, allowed * games_per_pair


def run(args):
    if args.top < 1 or not 1 <= args.workers <= 6:
        raise ValueError('--top must be positive and --workers must be 1..6')
    if args.games_per_pair < 6 or args.games_per_pair % 6:
        raise ValueError('--games-per-pair must be a positive multiple of 6')
    base_root = args.base_root.resolve()
    pool_v7 = reference_pool.load_pool(args.pool.resolve(), verify=True)
    game = Path(pool_v7['game']['path']).resolve()
    if sha256_file(game) != pool_v7['game']['sha256']:
        raise ValueError(f'{game} does not match pool v7 game sha256')
    args.out.mkdir(parents=True, exist_ok=True)
    base_out = base_root / 'sweeps/anchors' / args.out.name
    runs, before = fit_ladder(base_root)
    roster, skipped = select_roster(args.top, pool_v7, runs, before, base_out)
    print_roster(roster, skipped)
    groups = writer_groups(roster)
    allowed_pairs, skipped_pairs, total = schedule_counts(roster, args.games_per_pair)
    existing = load_games(args.out / 'games.jsonl')
    print(f'families={{{", ".join(f"{key}: {len(value)}" for key, value in sorted(groups.items()))}}} '
          f'allowed_pairs={allowed_pairs} skipped_pairs={skipped_pairs} '
          f'games_existing={len(existing)} games_total={total}', flush=True)
    started = time.monotonic()
    context = multiprocessing.get_context('spawn')
    for family in sorted(groups):
        process = context.Process(target=play_family, args=(roster, family, pool_v7, args, game))
        process.start()
        process.join()
        if process.exitcode:
            raise RuntimeError(f'writer family {family} failed with exit code {process.exitcode}')
    games = load_games(args.out / 'games.jsonl')
    if len(games) != total:
        raise RuntimeError(f'expected {total} compatible games, found {len(games)}')
    tournament = write_pool(args.out, roster, games, pool_v7, game, args.games_per_pair,
                            skipped_pairs, time.monotonic() - started)
    _, after = fit_ladder(base_root, args.out / 'tournament.json')
    table = comparison(roster, before, after, tournament)
    report = dict(roster=len(roster), skipped_missing=skipped, allowed_pairs=allowed_pairs,
                  skipped_pairs=skipped_pairs, games=len(games), elapsed=tournament['elapsed'],
                  issues=after['issues'], comparison=table)
    reference_pool.atomic_json(args.out / 'report.json', report)
    print(json.dumps(report, indent=1), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top', type=int, default=60)
    parser.add_argument('--games-per-pair', type=int, default=24)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--base-root', type=Path, default=BASE_ROOT)
    parser.add_argument('--pool', type=Path, default=POOL_V7)
    parser.add_argument('--library', type=Path, default=POLICY_LIBRARY)
    parser.add_argument('--research-library', type=Path, default=RESEARCH_LIBRARY)
    parser.add_argument('--out', type=Path, default=ROOT / f"native/artifacts/rl/sweeps/anchors/tournament-{time.strftime('%Y%m%d')}")
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
