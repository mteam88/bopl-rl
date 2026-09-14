"""Joint Bradley-Terry ladder over every rated game sitting on disk.

Sources, all on one scale with random = 0:
  - each pool.json `matches[]` (the pool's own round robin)
  - each `<run>/reference/**/*.json` `records` (candidate vs that file's pool)
  - human games (Matthew), excluded rounds dropped

Identity: checkpoints are sha256, scripted bots are their name, `random` and
`human` are those strings. A pool member `o7-041-final` and the run checkpoint
with the same sha256 are the same node.

Regularization is one virtual draw per played pair (see elo_ladder.bradley_terry),
not a fraction of the pair's games: 480-game matches keep their weight, 6-game
matches stay finite. Wald 95% CIs come from the observed information.

This module only reads artifacts. It never plays games.
"""
import json
import os
import re
import time
from pathlib import Path

import elo_ladder
import reference_pool

SKIP_DIRS = frozenset({
    'venv', '__pycache__', 'checkpoints', 'evaluations', 'reference', 'workers',
    '.git', 'node_modules', 'target', 'human',
})
SHA256 = re.compile(r'^[0-9a-f]{64}$')
SHA8 = re.compile(r'^[0-9a-f]{8}$')
ANCHOR = 'random'
HUMAN = 'human'


def stamp(path):
    try:
        value = Path(path).stat()
    except (FileNotFoundError, OSError):
        return None
    return value.st_ino, value.st_mtime_ns, value.st_size


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return default


def is_run_dir(path):
    path = Path(path)
    return (path / 'metrics.jsonl').is_file() or (path / 'checkpoints').is_dir()


def run_name(rel):
    """Stable dashboard id: sweep trials `overnight-7-041`, nested `long-run-1b-007-…`."""
    parts = Path(rel).parts
    if not parts:
        return 'run'
    if parts[0] == 'sweeps' and len(parts) >= 3:
        return f'{parts[1]}-{parts[2]}'
    if len(parts) == 1:
        return parts[0]
    return '-'.join(parts)


def family_of(rel):
    parts = Path(rel).parts
    if parts and parts[0] == 'sweeps' and len(parts) >= 2:
        return parts[1]
    if len(parts) >= 2:
        return parts[0]
    return parts[0] if parts else 'run'


def sweep_of(rel):
    parts = Path(rel).parts
    if parts and parts[0] == 'sweeps' and len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def discover_runs(root, extra=None):
    """Every directory under root with metrics.jsonl or checkpoints/, plus extras.

    extra: {name: Path} isolated workspaces (the old --run-root list).
    Returns {name: info} where info has path, rel, family, sweep, trial.
    """
    root = Path(root).resolve()
    found = {}
    used = set()

    def add(name, path, rel, family, sweep, trial, extra_flag=False):
        path = Path(path).resolve()
        key = name
        n = 2
        while key in found and found[key]['path'] != path:
            key = f'{name}-{n}'
            n += 1
        found[key] = dict(
            name=key, path=path, rel=str(rel), family=family,
            sweep=sweep, trial=trial, extra=extra_flag,
        )
        used.add(path)

    if root.is_dir():
        for dirpath, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            path = Path(dirpath)
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == 'sweeps' and len(rel.parts) == 2 and rel.parts[1] == 'anchors':
                dirs.clear()
                continue
            if not ((path / 'metrics.jsonl').is_file() or (path / 'checkpoints').is_dir()):
                continue
            dirs.clear()
            sweep, trial = sweep_of(rel)
            add(run_name(rel), path, rel, family_of(rel), sweep, trial)

    for name, path in (extra or {}).items():
        path = Path(path).resolve()
        if path in used:
            for row in found.values():
                if row['path'] == path:
                    row['name'] = name
                    break
            continue
        rel = path.name
        try:
            rel = path.relative_to(root)
        except ValueError:
            pass
        sweep, trial = sweep_of(rel) if not isinstance(rel, str) else (None, None)
        add(name, path, rel, family_of(rel) if not isinstance(rel, str) else name,
            sweep, trial, extra_flag=True)
    return found


def discover_sweeps(root):
    """{sweep_name: directory} for every folder with sweep.json under root/sweeps."""
    root = Path(root).resolve()
    found = {}
    sweeps = root / 'sweeps'
    if not sweeps.is_dir():
        return found
    for child in sorted(sweeps.iterdir()):
        if child.is_dir() and (child / 'sweep.json').is_file() and child.name != 'anchors':
            found[child.name] = child
    return found


def discover_pools(root, extra=None):
    """Load every pool.json under sweeps/anchors (verify off: this is a reader)."""
    root = Path(root).resolve()
    paths = []
    anchors = root / 'sweeps' / 'anchors'
    if anchors.is_dir():
        for child in sorted(anchors.iterdir()):
            file = child / 'pool.json'
            if file.is_file():
                paths.append(file)
    for path in extra or ():
        path = Path(path)
        if path.is_file() and path not in paths:
            paths.append(path)
    pools = []
    issues = []
    for path in paths:
        try:
            pool = reference_pool.load_pool(path, verify=False)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            issues.append(f'pool {path}: {error}')
            continue
        label = pool_label(path)
        pool['label'] = label
        pool['sha8'] = reference_pool.pool_sha8(pool)
        pools.append(pool)
    return pools, issues


def pool_label(path):
    name = Path(path).parent.name
    match = re.fullmatch(r'pool-v(.+)', name)
    if match:
        return f'v{match.group(1)}'
    return name


def member_id(member):
    kind = member.get('kind')
    if kind == 'random' or member.get('name') == ANCHOR:
        return ANCHOR
    if kind == 'scripted':
        return member['name']
    sha = member.get('sha256')
    if sha:
        return sha
    return member['name']


def add_record(records, a, b, wins_a, wins_b, draws):
    if not a or not b or a == b:
        return
    wins_a, wins_b, draws = int(wins_a or 0), int(wins_b or 0), int(draws or 0)
    if wins_a + wins_b + draws <= 0:
        return
    if (b, a) in records:
        row = records[(b, a)]
        row[0] += wins_b
        row[1] += wins_a
        row[2] += draws
        return
    row = records.setdefault((a, b), [0, 0, 0])
    row[0] += wins_a
    row[1] += wins_b
    row[2] += draws


def _touch_participant(participants, pid, **fields):
    row = participants.get(pid)
    if row is None:
        row = dict(id=pid, kind=fields.get('kind') or 'unknown', aliases=[],
                   name=None, sha256=None, run=None, steps=None, path=None)
        participants[pid] = row
    aliases = fields.pop('aliases', None) or []
    alias = fields.pop('alias', None)
    if alias:
        aliases = [*aliases, alias]
    for value in aliases:
        if value and value not in row['aliases']:
            row['aliases'].append(value)
    for key, value in fields.items():
        if value in (None, '', []):
            continue
        if key == 'kind':
            if row['kind'] in ('unknown', None) or value == 'checkpoint':
                row['kind'] = value
            continue
        if row.get(key) in (None, ''):
            row[key] = value


def rating_files(run_dir):
    """(path, payload) for every readable rating JSON under the run, sha8 preferring legacy."""
    run_dir = Path(run_dir)
    reference = run_dir / 'reference'
    if not reference.is_dir():
        return []
    grouped = {}
    for dirpath, dirs, files in os.walk(reference, followlinks=False):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        folder = Path(dirpath)
        for name in files:
            if not name.endswith('.json') or name.endswith('.tmp'):
                continue
            path = folder / name
            payload = read_json(path)
            if not payload or not isinstance(payload, dict):
                continue
            if payload.get('schema') not in (None, 1, reference_pool.SCHEMA):
                continue
            records = payload.get('records')
            ident = payload.get('checkpoint_identity') or {}
            sha = ident.get('sha256')
            pool_sha = payload.get('pool_sha256')
            if not records or not sha:
                continue
            games = payload.get('games') or payload.get('budget') or 0
            source = payload.get('source') or ''
            if source == 'evaluation':
                # In-run evaluation fits: ~144 games with no band matching, mostly
                # against the scripted training roster the learner exploits. In a
                # joint fit they rate a mid-run checkpoint 900+ through the bots
                # alone (the pool-v5 Goodhart, again). Only band ratings and pool
                # round-robins carry transitive information; leave these to the
                # run page's dashed in-run curve.
                continue
            key = (sha, pool_sha)
            previous = grouped.get(key)
            rank = (1, games if isinstance(games, (int, float)) else 0)
            if previous and previous[0] > rank:
                continue
            grouped[key] = (rank, path, payload)
    return [(path, payload) for _, path, payload in grouped.values()]


def ingest_pools(pools, records, participants, aliases):
    issues = []
    for pool in pools:
        label = pool.get('label') or pool_label(pool.get('path', ''))
        for member in pool.get('members') or ():
            pid = member_id(member)
            aliases[member['name']] = pid
            kind = 'random' if pid == ANCHOR else ('scripted' if member.get('kind') == 'scripted' else 'checkpoint')
            _touch_participant(
                participants, pid, kind=kind, name=member['name'],
                sha256=member.get('sha256'),
                aliases=[member['name']] + ([f'{label}:{member["name"]}'] if label else []),
            )
        for match in pool.get('matches') or ():
            a_name, b_name = match.get('a'), match.get('b')
            if a_name not in aliases or b_name not in aliases:
                issues.append(f'{label}: match {a_name} vs {b_name} names missing from members')
                continue
            add_record(records, aliases[a_name], aliases[b_name],
                       match.get('wins_a'), match.get('wins_b'), match.get('draws'))
    return issues


def ingest_ratings(runs, pools, records, participants, aliases, sha_index):
    issues = []
    by_sha = {pool.get('pool_sha256'): pool for pool in pools}
    by_sha8 = {pool.get('sha8') or pool['pool_sha256'][:8]: pool for pool in pools}
    n_files = 0
    for run in runs.values():
        for path, payload in rating_files(run['path']):
            n_files += 1
            ident = payload.get('checkpoint_identity') or {}
            sha = ident.get('sha256')
            if not sha:
                issues.append(f'{path}: rating missing checkpoint sha256')
                continue
            pool_sha = payload.get('pool_sha256')
            pool = by_sha.get(pool_sha) or by_sha8.get((pool_sha or '')[:8])
            opponent_ids = dict(aliases)
            if pool:
                for member in pool.get('members') or ():
                    opponent_ids[member['name']] = member_id(member)
            elif pool_sha:
                issues.append(f'{path}: unknown pool {(pool_sha or "")[:8]}')
            steps = payload.get('learner_steps')
            if steps is None:
                stem = Path(payload.get('checkpoint') or path.stem).stem
                try:
                    steps = int(stem.split('-', 1)[1])
                except (ValueError, IndexError):
                    steps = None
            _touch_participant(
                participants, sha, kind='checkpoint', sha256=sha, run=run['name'],
                steps=steps, path=str(run['path']),
                alias=f"{run['name']}@{steps}" if steps is not None else run['name'],
            )
            sha_index.setdefault(sha, []).append(run['name'])
            for opponent, row in (payload.get('records') or {}).items():
                oid = opponent_ids.get(opponent, opponent)
                if SHA256.match(str(oid) if oid else ''):
                    _touch_participant(participants, oid, kind='checkpoint', sha256=oid, alias=opponent)
                elif oid == ANCHOR:
                    _touch_participant(participants, ANCHOR, kind='random', name=ANCHOR)
                else:
                    _touch_participant(participants, oid, kind='scripted', name=oid, alias=opponent)
                add_record(records, sha, oid, row.get('wins'), row.get('losses'), row.get('draws'))
    return n_files, issues


def ingest_human(human_dir, records, participants, aliases, sha_index):
    issues = []
    human_dir = Path(human_dir) if human_dir else None
    if not human_dir or not human_dir.is_dir():
        return [], issues
    skip = reference_pool.excluded_rounds(human_dir / 'excluded.json')
    _touch_participant(participants, HUMAN, kind='human', name='Matthew', alias='human')
    per_opponent = {}
    skipped = 0
    used = 0
    for row in reference_pool.load_human_rows(human_dir):
        if (row.get('session'), row.get('round')) in skip:
            continue
        opponent = human_opponent_id(row, aliases)
        if not opponent:
            skipped += 1
            continue
        outcome = row.get('outcome', 'draw')
        if 'agent_won' in row and 'outcome' not in row:
            if row.get('draw'):
                outcome = 'draw'
            elif row.get('agent_won'):
                outcome = 'b'
            else:
                outcome = 'a'
        wins = losses = draws = 0
        if outcome == 'a':
            wins = 1
        elif outcome == 'b':
            losses = 1
        else:
            draws = 1
        if SHA256.match(opponent):
            _touch_participant(participants, opponent, kind='checkpoint', sha256=opponent)
        elif opponent == ANCHOR:
            _touch_participant(participants, ANCHOR, kind='random', name=ANCHOR)
        else:
            _touch_participant(participants, opponent, kind='scripted', name=opponent, alias=opponent)
        add_record(records, HUMAN, opponent, wins, losses, draws)
        slot = per_opponent.setdefault(opponent, [0, 0, 0])
        slot[0] += wins
        slot[1] += losses
        slot[2] += draws
        used += 1
    human_records = []
    for opponent, (wins, losses, draws) in per_opponent.items():
        person = participants.get(opponent) or dict(id=opponent, name=opponent[:12], aliases=[])
        human_records.append(dict(
            id=opponent, name=display_name(person),
            wins=wins, losses=losses, draws=draws, games=wins + losses + draws,
        ))
    human_records.sort(key=lambda row: -row['games'])
    if skipped:
        issues.append(f'human: {skipped} rounds against unknown opponents (kept {used})')
    return human_records, issues


def human_opponent_id(row, aliases):
    sha = row.get('agent_sha256')
    if sha:
        return sha
    b = row.get('b') or ''
    if isinstance(b, str) and b.startswith('policy:') and SHA256.match(b[7:]):
        return b[7:]
    if isinstance(b, str) and b.startswith('bot:'):
        name = b[4:]
        return aliases.get(name, name)
    agent = row.get('agent')
    if isinstance(agent, str) and agent.startswith('bot:'):
        name = agent[4:]
        return aliases.get(name, name)
    if isinstance(agent, str) and agent in aliases:
        return aliases[agent]
    if isinstance(b, str) and b in aliases:
        return aliases[b]
    return None


def tally(names, records):
    counts = {name: [0, 0, 0] for name in names}
    for (a, b), (wa, wb, d) in records.items():
        if a in counts:
            counts[a][0] += wa
            counts[a][1] += wb
            counts[a][2] += d
        if b in counts:
            counts[b][0] += wb
            counts[b][1] += wa
            counts[b][2] += d
    return counts


def display_name(row):
    if row.get('kind') == 'human':
        return row.get('name') or 'Matthew'
    if row.get('kind') == 'checkpoint' and row.get('run'):
        steps = row.get('steps')
        if steps is not None:
            return f"{row['run']} @ {int(steps):,}"
        return row['run']
    if row.get('name'):
        return row['name']
    aliases = row.get('aliases') or []
    if aliases:
        return aliases[0]
    sha = row.get('sha256') or row.get('id') or ''
    return sha[:12] if sha else '?'


def collect(root, runs, pools, human_dir):
    """Build the pair table. Does not fit."""
    records = {}
    participants = {}
    aliases = {ANCHOR: ANCHOR}
    sha_index = {}
    _touch_participant(participants, ANCHOR, kind='random', name=ANCHOR)
    issues = []
    issues.extend(ingest_pools(pools, records, participants, aliases))
    n_files, rating_issues = ingest_ratings(runs, pools, records, participants, aliases, sha_index)
    issues.extend(rating_issues)
    human_records, human_issues = ingest_human(human_dir, records, participants, aliases, sha_index)
    issues.extend(human_issues)
    present = {ANCHOR}
    for a, b in records:
        present.add(a)
        present.add(b)
    names = [pid for pid in participants if pid in present]
    return dict(
        records=records, participants=participants, names=names,
        human_records=human_records, issues=issues, rating_files=n_files,
        sha_index=sha_index, aliases=aliases,
    )


def fit_collection(collection):
    started = time.perf_counter()
    names = collection['names']
    fitted = elo_ladder.bradley_terry_fit(names, collection['records'], ANCHOR)
    counts = tally(names, collection['records'])
    elapsed = time.perf_counter() - started
    games = sum(sum(row) for row in collection['records'].values())
    rows = []
    for pid in names:
        person = collection['participants'][pid]
        wins, losses, draws = counts[pid]
        rows.append(dict(
            id=pid, name=display_name(person), kind=person.get('kind'),
            rating=fitted['ratings'][pid], interval=fitted['intervals'][pid],
            games=wins + losses + draws, wins=wins, losses=losses, draws=draws,
            run=person.get('run'), steps=person.get('steps'),
            sha256=person.get('sha256') or (pid if SHA256.match(pid) else None),
            aliases=person.get('aliases') or [],
        ))
    rows.sort(key=lambda row: (-(row['rating'] if row['rating'] is not None else -1e9), row['name']))
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
    human = next((row for row in rows if row['kind'] == 'human' or row['id'] == HUMAN), None)
    if human:
        named = []
        for rec in collection['human_records']:
            person = collection['participants'].get(rec['id'])
            named.append(dict(rec, name=display_name(person) if person else rec['name']))
        human = dict(human, records=named)
    return dict(
        rows=rows, human=human, issues=collection['issues'],
        ratings=fitted['ratings'], intervals=fitted['intervals'],
        stats=dict(
            participants=len(names), pairs=len(collection['records']),
            games=games, rating_files=collection['rating_files'],
            elapsed=elapsed, iterations=fitted['iterations'],
            virtual_draw=0.5, anchor=ANCHOR, ci=fitted.get('ci', 'wald'),
        ),
        sha_index=collection['sha_index'],
    )


def source_signature(root, runs, pools, human_dir):
    parts = []
    for pool in pools:
        parts.append(('pool', pool.get('path'), stamp(pool.get('path'))))
    for run in sorted(runs.values(), key=lambda row: row['name']):
        reference = Path(run['path']) / 'reference'
        files = []
        if reference.is_dir():
            for dirpath, dirs, filenames in os.walk(reference, followlinks=False):
                dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
                for name in filenames:
                    if name.endswith('.json'):
                        path = Path(dirpath) / name
                        files.append((str(path), stamp(path)))
        files.sort()
        parts.append(('run', run['name'], tuple(files)))
    if human_dir:
        human_dir = Path(human_dir)
        parts.append(('human', stamp(human_dir / 'games.jsonl'), stamp(human_dir / 'excluded.json')))
        parts.append(('human-sessions', tuple(sorted(
            (path.name, stamp(path)) for path in human_dir.glob('.session-*.jsonl')
        ))))
    return tuple(parts)


def build(root, runs, pools, human_dir):
    collection = collect(root, runs, pools, human_dir)
    return fit_collection(collection)
