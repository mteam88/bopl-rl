"""The league board: the payoff matrix, one agent, a live run, and a match.

Four pages, served as one no-build single-page app from `web/`.

  /            the league    the all-pairs payoff matrix over current members
  /agent/<id>  one member    its row, its rating history, who exploits it
  /run/<name>  a live run    training metrics and the opponents it actually saw
  /watch       a match       the replay arena

The league is the centre: a durable population that runs join and are measured
against. Everything is read from `native/rl/league/` (members and aggregated
payoff counts) and `native/artifacts/rl/` (per-run training artifacts).

Counts are never pooled across `condition` values; see payoff.py.
"""
import argparse
from collections import OrderedDict, defaultdict
import gzip
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import threading
import time
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse, parse_qs

from metrics_store import store
import packed
import payoff
import map_catalogue
import visuals

HERE = Path(__file__).resolve().parent

CHART_KEYS = (
    'step', 'learner_steps', 'elapsed', 'sps', 'learner_sps', 'policy_loss',
    'value_loss', 'entropy', 'kl', 'clip_fraction', 'gradient_norm',
    'explained_variance', 'mean_reward', 'entropy_coefficient', 'learning_rate',
    'advantage_std', 'entropy_aim', 'entropy_buttons', 'cuda_memory_mb',
    'win_rate', 'train_win_rate',
)
CHART_POINTS = 360
# An exploiter is called "arrived" the first time it holds these scores against
# its target; the pair is the shape of the curve, not just its end.
EXPLOIT_MARKS = (0.55, 0.70)


# ---------------------------------------------------------------- basics

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return default


def stamp(path):
    try:
        value = Path(path).stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return value.st_ino, value.st_mtime_ns, value.st_size


def encode_json(data):
    raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
    return raw, gzip.compress(raw, 1), '"' + hashlib.sha256(raw).hexdigest() + '"'


def round_json(value, ndigits=3):
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_json(item, ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_json(item, ndigits) for item in value]
    return value


# ------------------------------------------------- replay encoding (arena)
# The arena reads a replay as NDJSON: a header line, then one line per frame,
# so it can draw the first frame before the last has arrived. Unchanged.

def visual_ndjson(data):
    frames = data.pop('frames', [])
    chunks = [json.dumps(round_json(data), allow_nan=False, separators=(',', ':')).encode()]
    for frame in frames:
        chunks.append(json.dumps(round_json(frame), allow_nan=False, separators=(',', ':')).encode())
    return b'\n'.join(chunks) + b'\n'


def is_evaluation_visual(path):
    return (path.suffix == '.json' and path.name != 'result.json'
            and any(part.endswith('evaluations') for part in path.parts))


def gzip_header_name(blob):
    if len(blob) < 10 or blob[:2] != b'\x1f\x8b':
        return ''
    flags = blob[3]
    pos = 10
    if flags & 4:
        if len(blob) < 12:
            return ''
        pos = 12 + int.from_bytes(blob[10:12], 'little')
    if flags & 8:
        end = blob.find(b'\x00', pos)
        if end < 0:
            return ''
        return blob[pos:end].decode('latin-1')
    return ''


def gzip_named(raw, name, level=4):
    buf = io.BytesIO()
    with gzip.GzipFile(filename=name, mode='wb', fileobj=buf, compresslevel=level, mtime=0) as handle:
        handle.write(raw)
    return buf.getvalue()


def source_stamp_name(stat):
    return f'{stat.st_ino}-{stat.st_mtime_ns}-{stat.st_size}'


_visual_mem = OrderedDict()
_visual_mem_lock = threading.Lock()
_visual_path_locks = {}
_VISUAL_MEM = 16


def _visual_path_lock(path):
    key = str(path)
    with _visual_mem_lock:
        lock = _visual_path_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _visual_path_locks[key] = lock
        return lock


def encoded_visual(path):
    path = Path(path)
    # Finished runs are archived to name.json.gz (compact_evaluations.py); the
    # sidecar keeps the .json name so the report's reference stays valid.
    origin = visuals.source(path)
    if origin is None:
        raise FileNotFoundError(path)
    stat = origin.stat()
    key = source_stamp_name(stat)
    with _visual_mem_lock:
        hit = _visual_mem.get(str(path))
        if hit and hit[0] == key:
            _visual_mem.move_to_end(str(path))
            return hit[1], hit[2]
    sidecar = path.with_name(path.name + '.dash.gz')
    encoded = None
    with _visual_path_lock(path):
        stat = origin.stat()
        key = source_stamp_name(stat)
        if sidecar.is_file():
            blob = sidecar.read_bytes()
            if gzip_header_name(blob) == key:
                encoded = blob
        if encoded is None:
            encoded = gzip_named(visual_ndjson(visuals.load(path)), key)
            tmp = sidecar.with_name(sidecar.name + f'.{os.getpid()}.tmp')
            try:
                tmp.write_bytes(encoded)
                os.replace(tmp, sidecar)
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    etag = '"' + hashlib.sha256(encoded).hexdigest()[:32] + '"'
    with _visual_mem_lock:
        _visual_mem[str(path)] = (key, encoded, etag)
        _visual_mem.move_to_end(str(path))
        while len(_visual_mem) > _VISUAL_MEM:
            _visual_mem.popitem(last=False)
    return encoded, etag


# ------------------------------------------------------------ evaluations

def step_order(path):
    match = re.search(r'(\d+)', path.parent.name if path.name == 'result.json' else path.stem)
    return (int(match.group(1)) if match else -1, str(path))


def evaluation_results(directory):
    root = Path(directory) / 'evaluations'
    if not root.is_dir():
        return []
    found = [child / 'result.json' for child in root.iterdir() if (child / 'result.json').is_file()]
    found.sort(key=step_order)
    return found


def compact_report(path):
    stat = path.stat()
    value = _compact_report(path, stat.st_ino, stat.st_mtime_ns, stat.st_size)
    if not value:
        return None
    return dict(value, games=[dict(game) for game in value['games']])


@lru_cache(maxsize=256)
def _compact_report(path, inode, modified, size):
    report = json.loads(path.read_text())
    if not report:
        return None
    report = map_catalogue.normalize_record(report, *map_catalogue.run_records(path.parents[2]))
    value = {key: report[key] for key in
             ('schema', 'step', 'suite', 'groups', 'frozen_opponents', 'flags', 'flag_metrics')
             if key in report}
    value['games'] = [{key: game[key] for key in
                       ('name', 'opponent', 'map', 'seed', 'side', 'save', 'win',
                        'timeout', 'undecided', 'kills', 'deaths', 'ticks',
                        'replay', 'visual', 'mind')
                       if key in game} for game in report.get('games', [])]
    return value


def attach_media(report, folder):
    for game in report.get('games') or ():
        video = folder / (game['name'] + '.mp4')
        if video.exists():
            game['video'] = video.name
        poster = video.with_suffix('.jpg')
        if poster.exists():
            game['poster'] = poster.name
        mind = folder / (game['name'] + '.mind.json.gz')
        if mind.exists():
            game['mind'] = mind.name
    return report


def evaluation_payload(result_file):
    report = compact_report(result_file)
    if not report:
        return None
    report['directory'] = result_file.parent.name
    return attach_media(report, result_file.parent)


def evaluation_record(result_file):
    """The cheap per-evaluation summary: step, and the draw-aware score."""
    report = compact_report(result_file)
    if not report:
        return {'directory': result_file.parent.name}
    # A report records `win`, `timeout` and (since evaluate.py @25b093f8, so on
    # no report written before today) `undecided`. Both non-wins are draws:
    # a timeout is nobody's win, and an undecided round is both sides dying on
    # the same tick. Losses are whatever is left.
    wins = draws = 0
    games = report.get('games') or ()
    for game in games:
        if game.get('timeout') or game.get('undecided'):
            draws += 1
        elif game.get('win'):
            wins += 1
    played = len(games)
    losses = played - wins - draws
    return {
        'directory': result_file.parent.name,
        'step': report.get('step'),
        'wins': wins, 'losses': losses, 'draws': draws,
        'score': ((wins + 0.5 * draws) / played) if played else None,
    }


# The policy's own estimate of how the game is going, per decision. OpenAI
# added a win-probability head to Five purely for introspection and it became
# the most useful debugging tool they had; the head is already written to
# `<game>.mind.json.gz`, so the timeline can simply show it.

@lru_cache(maxsize=512)
def _value_points(path, inode, modified, size):
    with gzip.open(path, 'rt') as handle:
        mind = json.load(handle)
    return tuple((decision['t'], round(float(decision['value']), 4))
                 for decision in mind.get('decisions') or ()
                 if decision.get('value') is not None)


def value_points(path):
    stat = path.stat()
    return list(_value_points(path, stat.st_ino, stat.st_mtime_ns, stat.st_size))


def value_traces(directory, game):
    """One game's value trace at every checkpoint that played it.

    Because evaluation games are named by map/seed/side, the same name is the
    same fixed situation at every step of training, so stacking the traces
    shows the prediction sharpening (or failing to).
    """
    traces = []
    for result in evaluation_results(directory):
        folder = result.parent
        mind = folder / (game + '.mind.json.gz')
        if not mind.is_file():
            continue
        report = compact_report(result) or {}
        match = next((item for item in report.get('games') or () if item.get('name') == game), None)
        try:
            points = value_points(mind)
        except (OSError, ValueError, KeyError):
            continue
        if not points:
            continue
        traces.append({
            'directory': folder.name,
            'step': report.get('step'),
            'points': points,
            'win': bool(match.get('win')) if match else None,
            'timeout': bool(match.get('timeout')) if match else None,
        })
    traces.sort(key=lambda item: item['step'] or 0)
    return {'game': game, 'traces': traces}


# ------------------------------------------------------------------- runs

# Directories that are large and never contain a run. Walking into them costs
# real time on an artifact tree this size.
SKIP_DIRS = {'evaluations', 'final-evaluations', 'checkpoints', 'pool', 'game-logs',
             'workers', 'reference', 'source', 'venv', 'validation', 'dependencies',
             '__pycache__'}
RUN_DEPTH = 4


def is_run_dir(path):
    """A run is a directory with metrics (either spelling) or checkpoints."""
    return packed.exists(path / 'metrics.jsonl') or (path / 'checkpoints').is_dir()


def discover_runs(root, extra):
    """Every run under `root`, nested trials included.

    Sweep trials sit several levels down (`sweeps/<family>/<trial>`), so a flat
    scan of `root` misses most of the tree. Names are the path relative to
    `root`, which keeps them unique and stable; they contain slashes, which is
    why routing matches the longest known name rather than splitting on the
    first one.
    """
    found = {}
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            children = sorted(current.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir() or child.is_symlink() or child.name in SKIP_DIRS:
                continue
            if is_run_dir(child):
                found[child.relative_to(root).as_posix()] = child
            elif depth + 1 < RUN_DEPTH:
                stack.append((child, depth + 1))
    found.update(extra)
    return found


def run_status(directory):
    status = dict(read_json(directory / 'status.json', {'status': 'starting'}))
    if status.get('status') == 'running':
        try:
            os.kill(status['pid'], 0)
        except (ProcessLookupError, KeyError, TypeError):
            status['status'] = 'interrupted'
    return status


def thin_metrics(snapshot):
    rows = snapshot['rows']
    if len(rows) > CHART_POINTS:
        stride = max(1, (len(rows) - 1) // (CHART_POINTS - 1))
        picked = rows[::stride]
        if picked[-1] is not rows[-1]:
            picked.append(rows[-1])
        rows = picked
    return [{key: row[key] for key in CHART_KEYS if key in row} for row in rows]


# The panel that did not exist. An earlier run's opponent pool quietly
# degenerated to its last few checkpoints: training win rate sat pinned at 0.50
# while evaluation against the initial policy climbed 0.58 -> 0.67, and nothing
# on the dashboard showed it.
#
# There is no per-match opponent record anywhere in the artifact tree, and the
# per-update counts live in `metrics.jsonl`, which runs to hundreds of
# megabytes. So this reads the three small files that hold the same story:
#   pool/state.json   ledger of cumulative games per snapshot ever pooled,
#                     the live pool, and what each bank is serving right now
#   pool/roster.json  the scripted bots, their Elo and their realized games
#   pool.jsonl        the pool's lifecycle: snapshot / armed / loaded / evicted
#
# Note the trainer does not do PFSP. Matchmaking is Elo-weighted, so "intended"
# here is the pool policy from config.json (uniform over whatever the banks
# hold, split against `scripted_share`) rather than an explicit weight vector.


def opponent_distribution(directory):
    """Which opponents the learner actually played, against what the pool intended."""
    directory = Path(directory)
    state = read_json(directory / 'pool' / 'state.json', {}) or {}
    roster = read_json(directory / 'pool' / 'roster.json', {}) or {}
    config = read_json(directory / 'config.json', {}) or {}
    policy = config.get('selfplay') or {}
    pool = state.get('pool') or {}
    if not state and not roster:
        return {'available': False,
                'reason': 'This run has no pool/state.json or pool/roster.json, so '
                          'which opponents it played was never recorded.'}

    live = {entry.get('path'): entry.get('elo')
            for entry in pool.get('pool') or [] if entry.get('path')}
    banks = pool.get('banks') or []
    serving = [bank.get('cur_opp_path') for bank in banks if bank.get('cur_opp_path')]

    entries = []
    for path, record in (state.get('ledger') or {}).items():
        games = int(record.get('games') or 0)
        wins = int(record.get('wins') or 0)
        entries.append({
            'id': Path(path).name, 'kind': 'snapshot',
            'step': record.get('step'), 'learner_steps': record.get('learner_steps'),
            'games': games, 'wins': wins,
            'winrate': (wins / games) if games else None,
            'elo': live.get(path), 'live': path in live, 'serving': path in serving,
        })
    for entry in roster.get('entries') or []:
        games = int(entry.get('games') or 0)
        wins = int(entry.get('wins') or 0)
        entries.append({
            'id': entry.get('name'), 'kind': 'scripted', 'step': None,
            'games': games, 'wins': wins,
            'winrate': (wins / games) if games else None,
            'elo': entry.get('elo'), 'live': True, 'serving': False,
        })
    total = sum(entry['games'] for entry in entries) or 1
    for entry in entries:
        entry['share'] = entry['games'] / total
    entries.sort(key=lambda entry: -entry['games'])

    played = [entry for entry in entries if entry['games']]
    snapshots = [entry for entry in entries if entry['kind'] == 'snapshot']
    scripted = [entry for entry in entries if entry['kind'] == 'scripted']
    live_snapshots = [entry for entry in snapshots if entry['live']]
    scripted_games = sum(entry['games'] for entry in scripted)

    events = defaultdict(int)
    for line in _pool_events(directory):
        events[line.get('event') or 'unknown'] += 1

    return {
        'available': True,
        'entries': entries,
        'distinct': len(played),
        'pooled': len(snapshots),
        'live': len(live_snapshots),
        # The degeneration signature: live pool members the learner never met.
        'live_unplayed': sum(1 for entry in live_snapshots if not entry['games']),
        'serving': [Path(path).name for path in serving],
        'distinct_serving': len(set(serving)),
        'banks': len(banks),
        'top_share': sum(entry['share'] for entry in played[:4]),
        # Lifetime realized against the *final* intended share. `scripted_share`
        # is adaptive (it tracks how close the learner is to even), so these two
        # are not the same kind of number and the UI labels them as such rather
        # than subtracting one from the other.
        'scripted_realized_lifetime': scripted_games / total,
        'scripted_intended_final': roster.get('share'),
        'max_size': pool.get('max_size') or policy.get('max_size'),
        'swap_winrate': pool.get('swap_winrate') or policy.get('swap_winrate'),
        'snapshot_interval': pool.get('snapshot_interval') or policy.get('snapshot_interval'),
        'events': dict(events),
        'timeline': _pool_timeline(directory),
    }


def _pool_events(directory):
    path = Path(directory) / 'pool.jsonl'
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _pool_timeline(directory):
    """Each swap, so the pool's turnover over training is visible as a series."""
    out = []
    for record in _pool_events(directory):
        if record.get('event') not in ('armed', 'loaded', 'evicted'):
            continue
        out.append({
            'event': record['event'], 'step': record.get('step'),
            'learner_steps': record.get('learner_steps'), 'bank': record.get('bank'),
            'winrate': record.get('winrate'), 'elo': record.get('elo'),
        })
    out.sort(key=lambda item: item.get('learner_steps') or 0)
    if len(out) > 600:
        stride = len(out) // 600 + 1
        out = out[::stride]
    return out


def run_payload(directory, name, members, rated=None):
    status = run_status(directory)
    config = map_catalogue.normalize_record(read_json(directory / 'config.json') or {}, status)
    reports = evaluation_results(directory)
    evaluations = [evaluation_record(path) for path in reports]
    joined = [member for member in members
              if (member.get('origin') or {}).get('run') == name]
    joined.sort(key=lambda member: (member.get('origin') or {}).get('step') or 0)
    rated = rated or {}
    return dict(
        name=name, config=config, status=status,
        evaluations=evaluations,
        opponents=opponent_distribution(directory),
        # The run's own checkpoints as league members, with the league rating
        # each one holds: this run's trajectory on the league's scale.
        members=[{'id': member['id'], 'name': member.get('name'),
                  'kind': member.get('kind'),
                  'step': (member.get('origin') or {}).get('step'),
                  **(rated.get(member['id']) or {})}
                 for member in joined],
    )


# ----------------------------------------------------------------- league

def member_index(members):
    return {member['id']: member for member in members}


def agent_payload(agent_id, members, rows, condition):
    """One member: its row of the matrix, its history, and who beats it."""
    table = payoff.summary(members, rows, condition)
    ids = table['ids']
    if agent_id not in ids:
        return None
    seat = ids.index(agent_id)
    me = table['members'][seat]
    by_id = {member['id']: member for member in table['members']}
    row = []
    for other, cell in zip(ids, table['cells'][seat]):
        if other == agent_id or not cell:
            continue
        opponent = by_id[other]
        row.append({
            'id': other, 'name': opponent.get('name'), 'kind': opponent.get('kind'),
            'score': cell['score'], 'games': cell['games'],
            'loadouts': opponent.get('loadouts'),
        })
    row.sort(key=lambda item: item['score'])

    # Rating history: the other checkpoints of the same origin run, in step
    # order. A member's own rating is one number; the run it came from is what
    # has a trajectory.
    origin = (me.get('origin') or {}).get('run')
    history = []
    if origin:
        for member in table['members']:
            if (member.get('origin') or {}).get('run') != origin:
                continue
            history.append({
                'id': member['id'], 'step': (member.get('origin') or {}).get('step'),
                'rating': member.get('rating'), 'nash_value': member.get('nash_value'),
                'score': member.get('score'),
            })
        history.sort(key=lambda item: item['step'] or 0)

    return {
        'condition': condition,
        'conditions': payoff.conditions(rows),
        'member': me,
        'row': row,
        'history': history,
        'exploiters': time_to_exploit(agent_id, table),
        'specialists': specialist_regret(agent_id, table),
        'mirrored': table.get('mirrored'),
    }


def time_to_exploit(agent_id, table):
    """How fast a fresh exploiter reaches 55% and 70% against this member.

    A caveat the UI repeats: `league.json` has no field naming the member an
    exploiter was trained against, so the target is *inferred*. The best signal
    available is `parent` — an exploiter is forked from the main it is meant to
    beat — and only when that is missing does this fall back to whichever
    member the exploiter played most, which an even schedule makes ambiguous.
    A `target` field on the member would remove the guess entirely.
    """
    ids = table['ids']
    seat = ids.index(agent_id)
    by_id = {member['id']: member for member in table['members']}
    runs = defaultdict(list)
    for other_seat, other in enumerate(ids):
        member = by_id[other]
        if member.get('kind') != 'exploiter':
            continue
        cell = table['cells'][other_seat][seat]
        if not cell:
            continue
        parent = member.get('parent')
        if parent in by_id:
            if parent != agent_id:
                continue
        else:
            played = [(peer_cell['games'], peer) for peer, peer_cell
                      in zip(ids, table['cells'][other_seat]) if peer_cell]
            if not played or max(played)[1] != agent_id:
                continue
        run = (member.get('origin') or {}).get('run') or member['id']
        runs[run].append({
            'id': other, 'name': member.get('name'),
            'step': (member.get('origin') or {}).get('step'),
            # cells[exploiter][target] is already the exploiter's score
            # against this member; it is not the member's own score.
            'score': cell['score'],
            'games': cell['games'],
        })
    out = []
    for run, points in sorted(runs.items()):
        points.sort(key=lambda item: item['step'] or 0)
        marks = {}
        for mark in EXPLOIT_MARKS:
            hit = next((point for point in points if point['score'] >= mark), None)
            marks[str(mark)] = hit['step'] if hit else None
        out.append({'run': run, 'points': points, 'marks': marks,
                    'best': max((point['score'] for point in points), default=None)})
    return {'runs': out, 'inferred': True,
            'basis': 'parent' if any(by_id[other].get('parent') in by_id
                                     for other in ids
                                     if by_id[other].get('kind') == 'exploiter')
                     else 'most-played'}


def specialist_regret(agent_id, table):
    """Per-kit regret: what this member gives up to each ability specialist.

    Regret is reported as `0.5 - score`, i.e. how far below an even split this
    member lands against a specialist in that kit. The kit label comes from the
    specialist's `loadouts` descriptor, which is the only place the contract
    records what a specialist specialises in.
    """
    ids = table['ids']
    seat = ids.index(agent_id)
    by_id = {member['id']: member for member in table['members']}
    out = []
    for other_seat, other in enumerate(ids):
        member = by_id[other]
        if member.get('kind') != 'specialist' or other == agent_id:
            continue
        cell = table['cells'][seat][other_seat]
        if not cell:
            continue
        out.append({
            'id': other, 'name': member.get('name'),
            'loadouts': member.get('loadouts'),
            'score': cell['score'], 'games': cell['games'],
            'regret': 0.5 - cell['score'],
        })
    out.sort(key=lambda item: -item['regret'])
    return out


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1] / 'artifacts/rl',
                        help='training artifacts (per-run directories)')
    parser.add_argument('--league', type=Path, default=HERE.parent / 'league',
                        help='league.json + payoff.jsonl')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8876)
    parser.add_argument('--run-root', action='append', default=[], metavar='NAME=PATH',
                        help='Read a live training run directory from an isolated workspace')
    args = parser.parse_args()
    root = args.root.resolve()
    league_dir = args.league.resolve()
    extra_runs = {}
    for value in args.run_root:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('run_root requires NAME=PATH')
        extra_runs[name] = Path(path).resolve()

    condition_default = None
    lock = threading.RLock()
    league_state = {'sig': None, 'members': [], 'rows': []}
    page_state = OrderedDict()
    run_state = OrderedDict()
    eval_state = OrderedDict()
    metrics_state = OrderedDict()
    catalog_state = [None, None]

    def league_sig():
        return (stamp(league_dir / 'league.json'), stamp(league_dir / 'payoff.jsonl'))

    def league_data():
        sig = league_sig()
        with lock:
            if league_state['sig'] == sig:
                return league_state['members'], league_state['rows']
        members, rows = payoff.read(league_dir)
        with lock:
            league_state.update(sig=sig, members=members, rows=rows)
        return members, rows

    def pick_condition(rows, wanted):
        available = payoff.conditions(rows)
        names = [item['condition'] for item in available]
        if wanted in names:
            return wanted, available
        return (names[0] if names else None), available

    def catalog(force=False):
        sig = (stamp(root), tuple(sorted((name, str(path)) for name, path in extra_runs.items())))
        with lock:
            if not force and catalog_state[0] == sig and catalog_state[1] is not None:
                return catalog_state[1]
        found = discover_runs(root, extra_runs)
        with lock:
            catalog_state[0], catalog_state[1] = sig, found
        return found

    def split_run(rest):
        """(directory, name, tail) — run names contain slashes, so match the
        longest known name rather than splitting on the first separator."""
        known = catalog()
        parts = unquote(rest).split('/')
        for size in range(len(parts), 0, -1):
            name = '/'.join(parts[:size])
            if name in known:
                return known[name], name, '/'.join(parts[size:])
        return None, None, None

    def cached(store_map, key, sig, build, limit):
        with lock:
            hit = store_map.pop(key, None)
            if hit and hit[0] == sig:
                store_map[key] = hit
                return hit[1]
        encoded = encode_json(build())
        with lock:
            store_map[key] = (sig, encoded)
            while len(store_map) > limit:
                store_map.popitem(last=False)
        return encoded

    def league_payload(condition):
        members, rows = league_data()
        chosen, available = pick_condition(rows, condition)
        if chosen is None:
            return {'condition': None, 'conditions': [], 'members': [], 'ids': [],
                    'cells': [], 'cycles': {'count': 0, 'edges': 0, 'top': []},
                    'saturation': payoff.saturation(members, [], []),
                    'coverage': 0.0, 'pairs': 0, 'games': 0, 'mirrored': False,
                    'empty': not members,
                    'runs': sorted(catalog())}
        table = payoff.summary(members, rows, chosen)
        table['conditions'] = available
        table['empty'] = not members
        table['runs'] = sorted(catalog())
        return table

    def league_encoded(condition):
        return cached(page_state, ('league', condition),
                      (league_sig(), tuple(sorted(catalog()))),
                      lambda: league_payload(condition), 8)

    def agent_encoded(agent_id, condition):
        members, rows = league_data()
        chosen, _ = pick_condition(rows, condition)
        payload = agent_payload(agent_id, members, rows, chosen) if chosen else None
        if payload is None:
            return None
        return cached(page_state, ('agent', agent_id, condition), league_sig(),
                      lambda: payload, 16)

    def runs_encoded():
        def build():
            members, _ = league_data()
            rows = []
            for name, directory in sorted(catalog().items()):
                status = run_status(directory)
                reports = evaluation_results(directory)
                rows.append({
                    'name': name,
                    'status': status.get('status'),
                    'steps': status.get('learner_steps'),
                    'evaluations': len(reports),
                    'members': sum(1 for member in members
                                   if (member.get('origin') or {}).get('run') == name),
                    'modified': (stamp(directory / 'status.json') or (0, 0, 0))[1],
                })
            rows.sort(key=lambda item: -item['modified'])
            return rows
        sig = tuple((name, stamp(path / 'status.json'), stamp(path / 'evaluations'))
                    for name, path in sorted(catalog().items()))
        return cached(page_state, ('runs',), sig, build, 8)

    def rated_members():
        """{id: {rating, nash_value, condition}} for the default condition."""
        members, rows = league_data()
        chosen, _ = pick_condition(rows, condition_default)
        if chosen is None:
            return {}
        table = payoff.summary(members, rows, chosen)
        return {member['id']: {'rating': member.get('rating'),
                               'interval': member.get('interval'),
                               'nash_value': member.get('nash_value'),
                               'condition': chosen}
                for member in table['members']}

    def run_encoded(directory, name):
        members, _ = league_data()
        sig = (stamp(directory / 'config.json'), stamp(directory / 'status.json'),
               stamp(directory / 'evaluations'), stamp(directory / 'pool'),
               stamp(directory / 'pool.jsonl'), league_sig())
        return cached(run_state, name, sig,
                      lambda: run_payload(directory, name, members, rated_members()), 16)

    def metrics_encoded(directory, name):
        def build():
            try:
                summary = store.read(directory / 'metrics.jsonl')
            except ValueError:
                raise ValueError('Invalid metrics record')
            records = map_catalogue.run_records(directory)
            summary['rows'] = [map_catalogue.normalize_record(row, *records)
                               for row in summary['rows']]
            matches = map_catalogue.translate_map_matches(summary['map_matches'], *records)
            return dict(metrics=thin_metrics(summary), metrics_total=summary['count'],
                        map_matches=matches, map_catalogue=map_catalogue.IDENTITY)
        return cached(metrics_state, name, stamp(directory / 'metrics.jsonl'), build, 4)

    def values_encoded(directory, name, game):
        sig = (stamp(directory / 'evaluations'),
               tuple(stamp(path) for path in evaluation_results(directory)))
        payload = cached(page_state, ('values', name, game), sig,
                         lambda: value_traces(directory, game), 16)
        return payload

    def eval_encoded(directory, name, folder):
        result = directory / 'evaluations' / folder / 'result.json'
        if not result.is_file():
            return None
        payload = evaluation_payload(result)
        if payload is None:
            return None
        return cached(eval_state, (name, folder), stamp(result), lambda: payload, 16)

    def warm():
        # Keep the league hot so the first request is never the slow one. The
        # catalog is only re-walked occasionally: discovery touches every
        # directory under the artifact root (hundreds of runs), and a nested
        # trial appearing does not change the root's mtime, so the cheap
        # stat-based signature cannot see it. Once a minute is soon enough.
        passes = 0
        while True:
            try:
                catalog(force=passes % 6 == 0)
                league_encoded(None)
            except Exception:
                pass
            passes += 1
            time.sleep(10)

    catalog(force=True)
    threading.Thread(target=warm, daemon=True).start()

    # Client-routed pages. `/runs/` is matched before this and is file serving,
    # not a page, so a '/run' prefix test here is unambiguous.
    PAGES = ('/', '/index.html', '/league')
    PAGE_PREFIXES = ('/agent', '/run', '/watch')
    ASSETS = ('/app.js', '/style.css', '/arena.js')

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *arguments):
            if len(arguments) < 2 or str(arguments[1]) not in ('200', '206', '304'):
                super().log_message(fmt, *arguments)

        def send_snapshot(self, encoded):
            raw, compressed, etag = encoded
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.end_headers()
                return
            payload = raw
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('ETag', etag)
            self.send_header('Vary', 'Accept-Encoding')
            if len(raw) > 16384 and 'gzip' in self.headers.get('Accept-Encoding', ''):
                payload = compressed
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def json(self, data):
            self.send_snapshot(encode_json(data))

        def api(self, path, query):
            condition = (query.get('condition') or [None])[0]
            if path == '/api/league':
                return self.send_snapshot(league_encoded(condition))
            if path == '/api/runs':
                return self.send_snapshot(runs_encoded())
            if path.startswith('/api/agent/'):
                encoded = agent_encoded(unquote(path[len('/api/agent/'):]), condition)
                return self.send_snapshot(encoded) if encoded else self.send_error(404)
            if path.startswith('/api/run/'):
                directory, name, tail = split_run(path[len('/api/run/'):])
                if directory is None:
                    return self.send_error(404)
                if tail == 'metrics':
                    try:
                        return self.send_snapshot(metrics_encoded(directory, name))
                    except ValueError:
                        return self.send_error(500, 'Invalid metrics record')
                if tail.startswith('values/'):
                    game = tail[len('values/'):]
                    if not re.fullmatch(r'[A-Za-z0-9_.-]+', game):
                        return self.send_error(404)
                    encoded = values_encoded(directory, name, game)
                    return self.send_snapshot(encoded) if encoded else self.send_error(404)
                if tail.startswith('evaluation/'):
                    folder = tail[len('evaluation/'):]
                    if not re.fullmatch(r'[A-Za-z0-9_.-]+', folder):
                        return self.send_error(404)
                    encoded = eval_encoded(directory, name, folder)
                    return self.send_snapshot(encoded) if encoded else self.send_error(404)
                if tail:
                    return self.send_error(404)
                return self.send_snapshot(run_encoded(directory, name))
            return self.send_error(404)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path.startswith('/api/'):
                return self.api(path, parse_qs(parsed.query))
            if path.startswith('/runs/'):
                directory, _, tail = split_run(path[len('/runs/'):])
                if directory is None or not tail:
                    return self.send_error(404)
                file = (directory / tail).resolve()
                if not file.is_relative_to(directory) or file.suffix not in (
                        '.json', '.jsonl', '.rep', '.mp4', '.bin', '.pt', '.jpg', '.gz'):
                    return self.send_error(403)
            elif path in ASSETS:
                file = HERE / 'web' / path[1:]
            elif path in PAGES or path.startswith(PAGE_PREFIXES):
                file = HERE / 'web' / 'index.html'
            else:
                return self.send_error(404)
            return self.send_file(file)

        def send_file(self, file):
            archived_visual = is_evaluation_visual(file) and not file.is_file() and visuals.exists(file)
            if not file.is_file() and not archived_visual:
                return self.send_error(404)
            if is_evaluation_visual(file) and (archived_visual or not self.headers.get('Range')):
                return self.send_visual(file)
            size = file.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                if not match or not any(match.groups()):
                    return self.send_error(416)
                a, b = match.groups()
                if a:
                    start = int(a)
                    end = min(end, int(b)) if b else end
                else:
                    start = max(0, size - int(b))
                if start > end or start >= size:
                    return self.send_error(416)
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type',
                             'application/json' if file.name.endswith('.json.gz')
                             else mimetypes.guess_type(file)[0] or 'application/octet-stream')
            if file.name.endswith('.json.gz'):
                self.send_header('Content-Encoding', 'gzip')
            if file.suffix in ('.html', '.js', '.css'):
                self.send_header('Cache-Control', 'no-cache')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(max(0, end - start + 1)))
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            with file.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

        def send_visual(self, file):
            encoded, etag = encoded_visual(file)
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.end_headers()
                return
            payload = encoded
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('ETag', etag)
            self.send_header('Vary', 'Accept-Encoding')
            if 'gzip' in self.headers.get('Accept-Encoding', ''):
                self.send_header('Content-Encoding', 'gzip')
            else:
                payload = gzip.decompress(encoded)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if self.command == 'HEAD':
                return
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_HEAD(self):
            return self.do_GET()

        def end_headers(self):
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Bopl league board: http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
