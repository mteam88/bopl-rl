"""Append cross-family v4 games using the per-player writer-ablation stack."""
import argparse
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

FIXED_ROOT = Path('/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/policy-v4-work')
FIXED_RESEARCH = FIXED_ROOT / 'native/rl/research'
FIXED_LIBRARY = FIXED_ROOT / 'native/artifacts/rl/libbopl_policy_eval.so'
sys.path.insert(0, str(FIXED_RESEARCH))

# Pin these names before tournament.py puts its own research directory first.
import layout  # noqa: E402,F401
import native_policy  # noqa: E402,F401
import environment  # noqa: E402
import elo_ladder  # noqa: E402

import tournament as base  # noqa: E402

_callback_spec = importlib.util.spec_from_file_location(
    'tournament_callback_ladder', Path(__file__).with_name('elo_ladder.py'))
callback_ladder = importlib.util.module_from_spec(_callback_spec)
_callback_spec.loader.exec_module(callback_ladder)


def cross_tasks(roster, maps, games_per_pair, existing):
    done = {base.task_key(row) for row in existing}
    tasks = []
    pairs = 0
    for index, a in enumerate(roster):
        a_writer = base.writer_ablation(a)
        if a_writer is None:
            continue
        for b in roster[index + 1:]:
            b_writer = base.writer_ablation(b)
            if b_writer is None or a_writer == b_writer:
                continue
            pairs += 1
            for task in elo_ladder.pair_tasks(a['name'], b['name'], maps,
                                              games_per_pair // len(maps), base.SEED_BASE):
                if base.task_key(task) not in done:
                    tasks.append(task)
    return pairs, tasks


def smoke(roster, pool, game, out):
    a = next(member for member in roster if base.writer_ablation(member) == 10)
    b = next(member for member in roster if base.writer_ablation(member) == 8)
    base.reference_pool.configure_games(out / 'workers/cross-smoke', game)
    models = {member['name']: elo_ladder.Model(member['name'], member['source'], 'checkpoint',
                                               library=FIXED_LIBRARY) for member in (a, b)}
    with environment.Environment(seed=89902, limit=int(pool['episode_ticks']),
                                 delay=int(pool.get('observation_delay_ticks', 9))) as env:
        env.reset(89902, 0)
        models[a['name']].attach(env, 0)
        models[b['name']].attach(env, 1)
        observed = [env.ablation_of(0), env.ablation_of(1)]
    expected = [10, 8]
    if observed != expected:
        raise RuntimeError(f'cross-family smoke writer bits {observed}, expected {expected}')
    games = elo_ladder.play_games(
        models, elo_ladder.pair_tasks(a['name'], b['name'], [0], 2, 89902),
        int(pool['episode_ticks']), 1, 89902, delay=int(pool.get('observation_delay_ticks', 9)))
    if len(games) != 2:
        raise RuntimeError(f'cross-family smoke completed {len(games)} games, expected 2')
    print(f"cross_smoke={a['name']}[10] vs {b['name']}[8] observed={observed} outcomes={[g['outcome'] for g in games]}",
          flush=True)


def play_shard(roster, pool, args, game, tasks, worker):
    os.environ.pop('BOPL4_ABLATE', None)
    os.environ.pop('BOPL_RESEARCH_LIBRARY', None)
    os.environ['BOPL_RL_BACKEND'] = 'rust'
    base.reference_pool.configure_games(args.out / f'workers/cross-{worker}', game)
    members = {member['name']: member for member in roster}
    players = base.playable(roster, FIXED_LIBRARY)
    descriptor = os.open(args.out / 'games.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        def persist(raw):
            row = base.game_row(raw, members)
            os.write(descriptor, (json.dumps(row, separators=(',', ':')) + '\n').encode())
        callback_ladder.play_games(players, tasks, int(pool['episode_ticks']), 1, base.SEED_BASE,
                                   delay=int(pool.get('observation_delay_ticks', 9)), on_result=persist)
    finally:
        os.close(descriptor)


def initial_comparison(partial, after, tournament):
    after_rows = {row['id']: row for row in after['rows']}
    local_rows = {base.joint_ladder.member_id(row): row for row in tournament['members']}
    rows = []
    for old in partial['comparison']:
        new = after_rows.get(old['id'])
        local = local_rows.get(old['id'])
        if not new or not local:
            continue
        rows.append(dict(member=old['member'], id=old['id'], rank_before=old['rank_before'],
                         rank_after=new['rank'], rating_before=old['rating_before'],
                         rating_tournament=local['rating'], rating_after=new['rating'],
                         ci_before=old['ci_before'], ci_after=new['interval'][1] - new['interval'][0],
                         tournament_interval=local['interval']))
    rows.sort(key=lambda row: row['rank_after'])
    return rows


def run(args):
    if not 1 <= args.workers <= 6:
        raise ValueError('--workers must be 1..6')
    partial_path = args.out / 'report.json'
    if not partial_path.is_file():
        raise RuntimeError('writer-family report.json is missing; let the current family run finish first')
    partial = json.loads(partial_path.read_text())
    pool = base.reference_pool.load_pool(args.pool.resolve(), verify=True)
    game = Path(pool['game']['path']).resolve()
    if base.sha256_file(game) != pool['game']['sha256']:
        raise RuntimeError('pool-v7 simulator SHA changed')
    partial_pool = json.loads((args.out / 'tournament.json').read_text())
    roster = []
    for member in partial_pool['members']:
        item = {key: member[key] for key in ('name', 'kind', 'sha256', 'metadata_sha256', 'path')
                if member.get(key) is not None}
        if item['kind'] == 'checkpoint':
            item['source'] = (args.base_out / item['path']).resolve()
        roster.append(item)
    skipped = int(partial.get('skipped_missing') or 0)
    smoke(roster, pool, game, args.out)
    existing = base.load_games(args.out / 'games.jsonl')
    pairs, tasks = cross_tasks(roster, pool['maps'], args.games_per_pair, existing)
    cross_roster = [member for member in roster if base.writer_ablation(member) is not None]
    print(f'cross_pairs={pairs} games_existing={len(existing)} cross_games_missing={len(tasks)}', flush=True)
    started = time.monotonic()
    context = multiprocessing.get_context('spawn')
    processes = []
    for worker in range(args.workers):
        process = context.Process(target=play_shard,
                                  args=(cross_roster, pool, args, game, tasks[worker::args.workers], worker))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode]
    if failed:
        raise RuntimeError(f'cross-family worker exits: {failed}')
    games = base.load_games(args.out / 'games.jsonl')
    full_pairs = len(roster) * (len(roster) - 1) // 2
    expected = full_pairs * args.games_per_pair
    if len(games) != expected:
        raise RuntimeError(f'expected {expected} full-round-robin games, found {len(games)}')
    elapsed = float(partial.get('elapsed') or 0) + time.monotonic() - started
    tournament = base.write_pool(args.out, roster, games, pool, game, args.games_per_pair, 0, elapsed)
    legacy_sha = base.sha256_file(base.POLICY_LIBRARY)
    fixed_sha = base.sha256_file(FIXED_LIBRARY)
    research_sha = base.sha256_file(FIXED_ROOT / 'native/artifacts/rl/libbopl_research.so')
    tournament['note'] = (
        'Full direct top-ladder round robin; side-swapped fixed-seed games on pool-v7 maps. '
        f'1,583 same-writer-family or writer-agnostic-bridge pairs used legacy policy evaluator {legacy_sha}; '
        f'1,267 cross-family v4 pairs used per-player writer-ablation policy evaluator {fixed_sha} '
        f'and research binding {research_sha} (policy-v4-work 6c19394). Local Bradley-Terry fit with '
        'random anchored at 0; Wald 95% intervals.')
    base.reference_pool.atomic_json(args.out / 'tournament.json', tournament)
    _, after = base.fit_ladder(args.base_root.resolve(), args.out / 'tournament.json')
    table = initial_comparison(partial, after, tournament)
    report = dict(roster=len(roster), skipped_missing=skipped, allowed_pairs=full_pairs,
                  skipped_pairs=0, games=len(games), elapsed=elapsed, issues=after['issues'], comparison=table)
    base.reference_pool.atomic_json(args.out / 'report.json', report)
    args.base_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.out / 'tournament.json', args.base_out / 'tournament.json')
    shutil.copy2(args.out / 'games.jsonl', args.base_out / 'games.jsonl')
    print(json.dumps(report, indent=1), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top', type=int, default=60)
    parser.add_argument('--games-per-pair', type=int, default=24)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--base-root', type=Path, default=base.BASE_ROOT)
    parser.add_argument('--pool', type=Path, default=base.POOL_V7)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-out', type=Path, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
