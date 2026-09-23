"""Quality dashboard: every run under --root, joint-fit Elo plus labelled pool scores."""
import argparse
from collections import OrderedDict
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
import traceback
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from metrics_store import store
from packed import resolved

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'sweep')]
import reference_pool
import joint_ladder
import lineup_report
import map_catalogue
import visuals

CHART_KEYS = (
    'step', 'learner_steps', 'elapsed', 'sps', 'learner_sps', 'policy_loss',
    'value_loss', 'entropy', 'kl', 'clip_fraction', 'gradient_norm',
    'explained_variance', 'mean_reward', 'entropy_coefficient', 'learning_rate',
    'advantage_std', 'entropy_aim', 'entropy_buttons', 'cuda_memory_mb',
)
CHART_POINTS = 360
LAST_K = 3
# A pool draw makes nearly every lineup unique, so a table of all of them is one
# row per game: keep the most-played ones and say how many were dropped.
LINEUP_ROWS = 32
LINEUP_COLUMNS = 16
# The sweep objective's pool (sweeps/anchors/pool-v7, sha8 d48f4b64); labels only, never the joint scale.
PRIMARY_POOL = 'v7'


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return default


def stamp(path):
    try:
        value = resolved(path).stat()
    except FileNotFoundError:
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
                       ('name', 'opponent', 'map', 'seed', 'side', 'save', 'win', 'loss',
                        'timeout', 'kills', 'deaths', 'ticks', 'replay', 'visual', 'mind')
                       if key in game} for game in report.get('games', [])]
    return value


def step_order(path):
    match = re.search(r'(\d+)', path.parent.name if path.name == 'result.json' else path.stem)
    return (int(match.group(1)) if match else -1, str(path))


def latest_report(directory, pattern):
    reports = sorted(directory.glob(pattern), key=step_order)
    return compact_report(reports[-1]) if reports else None


def evaluation_digest(path):
    """lineup_report.digest of one evaluation, cached per (path, stat).

    result.json carries every game of the evaluation, so a run page that wants
    per-ability and per-lineup counters over the whole training curve must never
    re-read the ones it already read: only the newest file costs a parse.
    """
    stat = path.stat()
    return _evaluation_digest(path, stat.st_ino, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2048)
def _evaluation_digest(path, inode, modified, size):
    return lineup_report.digest(json.loads(path.read_text()))


def run_digests(directory):
    rows = []
    for path in evaluation_results(directory):
        try:
            rows.append(evaluation_digest(path))
        except (ValueError, OSError):   # a result.json still being written
            continue
    return rows


def run_selection(directory, digests):
    """The run's loadout block: config.json, else what the last evaluation resolved."""
    config = read_json(directory / 'config.json', {}) or {}
    if config.get('loadouts'):
        return lineup_report.selection(config)
    return digests[-1]['selection'] if digests else lineup_report.selection(None)


def loadout_ratings(directory):
    """Per-lineup frozen-pool ratings, newest rated checkpoint of each rating directory.

    reference_pool writes a loadout selection's ratings under
    reference/<pool_sha8>/loadout-<loadout_sha8>/; a default-directory rating
    carries per-lineup rows only when it was itself played on a set.
    """
    root = Path(directory) / 'reference'
    found = []
    if not root.is_dir():
        return found
    for pool_dir in sorted(child for child in root.iterdir() if child.is_dir()):
        loadout_dirs = sorted(child for child in pool_dir.iterdir()
                              if child.is_dir() and child.name.startswith('loadout-'))
        for folder in (pool_dir, *loadout_dirs):
            best = None
            for path in folder.glob('policy-*.json'):
                payload = read_json(path, {}) or {}
                if not payload.get('lineups') or payload.get('rating') is None:
                    continue
                steps = reference_pool._checkpoint_learner_steps(directory, path.stem) or 0
                if best is None or steps > best[0]:
                    best = (steps, path, payload)
            if best is None:
                continue
            steps, path, payload = best
            found.append(dict(
                pool_sha8=pool_dir.name, loadout=None if folder is pool_dir else folder.name,
                checkpoint=path.stem + '.bin', learner_steps=steps, rating=payload.get('rating'),
                interval=payload.get('interval'), games=payload.get('games'),
                source=payload.get('source'),
                rows=[dict(lineup_report.lineup_entry(key), rating=row.get('rating'),
                           interval=row.get('interval'), games=row.get('games'), draws=row.get('draws'))
                      for key, row in sorted((payload.get('lineups') or {}).items())]))
    return found


def ability_trend(digests):
    """{ability id: [{step, per_game}]} — activations per game it was drawn in, over steps."""
    trend = {}
    for row in digests:
        for ability in row['abilities']:
            if not ability['games']:
                continue
            trend.setdefault(ability['id'], []).append(
                dict(step=row['step'], per_game=ability['activations'] / ability['games']))
    return trend


def lineup_trend(digests):
    """{lineup key: [{step, scripted, random, games}]} — the curve that shows a lineup stalling."""
    trend = {}
    for row in digests:
        for key, families in row['lineups'].items():
            point = dict(step=row['step'], games=sum(cell['games'] for cell in families.values()))
            for family in ('scripted', 'random'):
                cell = families.get(family)
                point[family] = cell['wins'] / cell['games'] if cell and cell['games'] else None
            trend.setdefault(key, []).append(point)
    return trend


def loadout_payload(directory, last_k=LAST_K):
    """The run page's `loadouts` / `abilities` / `lineups` keys."""
    digests = run_digests(directory)
    selection = run_selection(directory, digests)
    window = digests[-max(1, int(last_k)):]
    merged = lineup_report.merge_digests(window)
    abilities = ability_trend(digests)
    curves = lineup_trend(digests)
    ratings = loadout_ratings(directory)
    rated = {} if not ratings else {row['key']: row for row in ratings[-1]['rows']}
    default = ','.join(str(a) for a in lineup_report.DEFAULT_SELECTION['fixed'])
    table = lineup_report.lineup_table(merged['lineups'])
    kept = {row['key'] for row in sorted(table, key=lambda row: (row['key'] != default, -row['games']))[:LINEUP_ROWS]}
    rows = []
    for row in table:
        if row['key'] not in kept:
            continue
        hit = rated.get(row['key'])
        rows.append(dict(row, trend=curves.get(row['key']) or [],
                         rating=None if not hit else hit['rating'],
                         rating_interval=None if not hit else hit['interval'],
                         comparable=row['key'] == default))
    return dict(
        loadouts=dict(lineup_report.selection_brief(selection),
                      evaluations=len(digests), window=len(window), telemetry=merged['telemetry'],
                      steps=[row['step'] for row in digests],
                      timeout_rate=window[-1]['timeout_rate'] if window else None,
                      declared=bool((read_json(directory / 'config.json', {}) or {}).get('loadouts'))),
        abilities=[dict(row, trend=abilities.get(row['id']) or []) for row in merged['abilities']],
        lineups=dict(rows=rows, games=merged['games'], window=len(window), distinct=len(table),
                     hidden=len(table) - len(rows), ratings=ratings, rated=bool(rated)))


def discover_sweeps(root, extra_runs):
    found = dict(joint_ladder.discover_sweeps(root))
    for path in extra_runs.values():
        parent = Path(path).resolve().parent
        if (parent / 'sweep.json').is_file():
            found.setdefault(parent.name, parent)
    return found


def pool_brief(pool, members=False):
    if not pool:
        return None
    pool = map_catalogue.normalize_record(pool, pool)
    brief = dict(
        path=pool.get('path'), sha256=pool.get('pool_sha256'),
        sha8=pool.get('sha8') or reference_pool.pool_sha8(pool),
        label=pool.get('label') or joint_ladder.pool_label(pool.get('path') or ''),
        sim_sha256=(pool.get('game') or {}).get('sha256'),
        maps=pool.get('maps'),
    )
    if members:
        brief['members'] = pool.get('members') or []
    return brief


def loadout_brief(config):
    """The run list / sweep table badge: mode plus a tooltip, off config.json alone."""
    selection = lineup_report.selection(config)
    names = lambda ids: ' / '.join(item['name'] for item in lineup_report.named(ids))
    if selection['mode'] == 'set':
        title = f"{selection['set_file']} · {len(selection['lineups'])} lineups"
    elif selection['mode'] == 'pool':
        title = names(selection['pool'])
    else:
        title = names(selection['fixed'])
    short = dict(set=f"set ×{len(selection['lineups'])}", pool=f"pool {len(selection['pool'])}")
    return dict(mode=selection['mode'], badge=lineup_report.selection_badge(selection), title=title,
                short=short.get(selection['mode']) or lineup_report.selection_badge(selection),
                declared=bool((config or {}).get('loadouts')))


def config_bits(config):
    config = config or {}
    return dict(
        banks=config.get('banks'),
        scripted_banks=config.get('scripted_banks'),
        share_min=config.get('scripted_share_min'),
        share_max=config.get('scripted_share_max'),
    )


def sweep_pool_path(state):
    scoring = ((state or {}).get('config') or {}).get('scoring') or {}
    return scoring.get('pool')


def evaluation_results(directory):
    root = Path(directory) / 'evaluations'
    if not root.is_dir():
        return []
    found = [child / 'result.json' for child in root.iterdir() if (child / 'result.json').is_file()]
    found.sort(key=step_order)
    return found


def rating_signature(directory, pool=None):
    files = []
    root = Path(directory) / 'reference'
    if not root.is_dir():
        return tuple()
    for path in root.rglob('*.json'):
        files.append((str(path), stamp(path)))
    files.sort()
    return tuple(files)


def run_flags(directory):
    result = read_json(directory / 'result.json', {})
    if result.get('flags') is not None:
        return result['flags']
    reports = evaluation_results(directory)
    if reports:
        return read_json(reports[-1], {}).get('flags') or []
    return []


def run_summary(directory, pool, name, last_k=LAST_K):
    curve = reference_pool.rated_curve(directory, pool) if pool else []
    summary = None
    if curve:
        window = curve[-max(1, int(last_k)):]
        score, interval = reference_pool.iqm_interval(window)
        summary = dict(score=score, interval=interval, last_k=window,
                       game_mismatch=any(row.get('game_mismatch') for row in window))
    return dict(
        name=name, kind='run', path=str(directory),
        rating=None if not summary else summary['score'],
        interval=None if not summary else summary['interval'],
        spark=[{'steps': row['learner_steps'], 'rating': row['rating'],
                'lo': (row['interval'] or [None, None])[0],
                'hi': (row['interval'] or [None, None])[1]} for row in curve],
        ratings=curve,
        flags=run_flags(directory),
        game_mismatch=bool(summary and summary['game_mismatch']),
        unrated=summary is None,
        rate_command=None if summary else reference_pool.rate_command(directory, pool.get('path') if pool else None),
        last_k=None if not summary else [
            dict(checkpoint=row['checkpoint'], learner_steps=row['learner_steps'],
                 rating=row['rating'], interval=row['interval'], games=row.get('games'))
            for row in summary['last_k']],
    )


def spark_points(curve):
    return [{'steps': row['learner_steps'], 'rating': row['rating'],
             'lo': (row['interval'] or [None, None])[0],
             'hi': (row['interval'] or [None, None])[1]} for row in curve]


def pool_summaries(directory, name, pools, last_k=LAST_K):
    """One labelled frozen-pool curve per pool that has ratings on this run."""
    found = []
    for pool in pools or ():
        row = run_summary(directory, pool, name, last_k=last_k)
        if row['unrated'] and not row.get('ratings'):
            continue
        found.append(dict(
            label=pool.get('label') or joint_ladder.pool_label(pool.get('path') or ''),
            sha8=pool.get('sha8') or reference_pool.pool_sha8(pool),
            sha256=pool.get('pool_sha256'),
            rating=row['rating'], interval=row['interval'], spark=row['spark'],
            ratings=row['ratings'], unrated=row['unrated'],
            game_mismatch=row['game_mismatch'], last_k=row['last_k'],
            rate_command=row['rate_command'], flags=row['flags'],
        ))
    return found


def joint_for_run(fit, name):
    if not fit:
        return None
    rows = [row for row in fit.get('rows') or () if row.get('run') == name]
    if not rows:
        return None
    best = max(rows, key=lambda row: (row.get('games') or 0, row.get('steps') or 0))
    curve = [row for row in rows if row.get('steps') is not None]
    curve.sort(key=lambda row: row['steps'])
    return dict(
        rating=best['rating'], interval=best['interval'], games=best.get('games'),
        steps=best.get('steps'), id=best['id'],
        spark=[{'steps': row['steps'], 'rating': row['rating'],
                'lo': (row['interval'] or [None, None])[0],
                'hi': (row['interval'] or [None, None])[1]} for row in curve],
        ratings=[dict(checkpoint=row.get('name'), learner_steps=row.get('steps'),
                      rating=row['rating'], interval=row['interval'],
                      games=row.get('games'), source='joint', sha256=row.get('sha256'))
                 for row in curve],
    )


def human_for_run(fit, name):
    if not fit or not fit.get('human'):
        return None
    ids = {row['id'] for row in fit.get('rows') or () if row.get('run') == name}
    records = [row for row in (fit['human'].get('records') or ()) if row['id'] in ids]
    if not records:
        return None
    human = fit['human']
    return dict(rating=human['rating'], interval=human['interval'],
                games=human.get('games'), wins=human.get('wins'),
                losses=human.get('losses'), draws=human.get('draws'),
                records=records)


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


def thin_metrics(snapshot):
    rows = snapshot['rows']
    if len(rows) > CHART_POINTS:
        stride = max(1, (len(rows) - 1) // (CHART_POINTS - 1))
        picked = rows[::stride]
        if picked[-1] is not rows[-1]:
            picked.append(rows[-1])
        rows = picked
    return [{key: row[key] for key in CHART_KEYS if key in row} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE.parents[1] / 'artifacts/rl')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8876)
    parser.add_argument('--run-root', action='append', default=[], metavar='NAME=PATH',
                        help='Read a live training run directory from an isolated workspace')
    parser.add_argument('--pool', type=Path, default=None,
                        help='Primary frozen pool.json for labels (not the joint scale). '
                             f'Default: pool {PRIMARY_POOL} if present, else the last discovered pool')
    parser.add_argument('--human', type=Path, default=reference_pool.DEFAULT_HUMAN,
                        help='human/mac directory (games.jsonl + excluded.json)')
    args = parser.parse_args()
    root = args.root.resolve()
    extra_runs = {}
    for value in args.run_root:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('run_root requires NAME=PATH')
        extra_runs[name] = Path(path).resolve()

    cache_lock = threading.RLock()
    catalog_state = [None, None]
    pools_state = [None, None, None]
    summary_state = {}
    index_state = [None, None]
    matrix_state = [None, None]
    joint_state = dict(sig=None, encoded=None, fit=None, building=False, error=None)
    sweep_state = {}
    run_state = OrderedDict()
    eval_state = OrderedDict()
    metrics_state = OrderedDict()

    def load_pools():
        extra = [args.pool] if args.pool and args.pool.is_file() else []
        sig = tuple((str(path), stamp(path)) for path in extra)
        anchors = root / 'sweeps' / 'anchors'
        if anchors.is_dir():
            sig += tuple((str(child), stamp(child / 'pool.json'), stamp(child / 'tournament.json'))
                         for child in sorted(anchors.iterdir()) if child.is_dir())
        with cache_lock:
            if pools_state[0] == sig:
                return pools_state[1], pools_state[2]
        pools, issues = joint_ladder.discover_pools(root, extra=extra)
        primary = None
        if args.pool and args.pool.is_file():
            want = str(args.pool.resolve())
            primary = next((item for item in pools if item.get('path') == want), None)
        if primary is None:
            primary = next((item for item in pools if item.get('label') == PRIMARY_POOL), None)
        if primary is None and pools:
            primary = pools[-1]
        with cache_lock:
            pools_state[0], pools_state[1], pools_state[2] = sig, pools, primary
        return pools, primary

    def catalog(force=False):
        extras = dict(extra_runs)
        extra_sig = tuple((name, str(path), stamp(path)) for name, path in extras.items())
        with cache_lock:
            if not force and catalog_state[0] == extra_sig and catalog_state[1] is not None:
                return catalog_state[1]
        found = joint_ladder.discover_runs(root, extra=extras)
        with cache_lock:
            catalog_state[0], catalog_state[1] = extra_sig, found
        return found

    def run_directory(name):
        known = catalog()
        if name in known:
            return known[name]['path']
        if re.fullmatch(r'[A-Za-z0-9_.-]+', name) and (root / name / 'config.json').is_file():
            return root / name
        return None

    def split_sweep(name, info=None):
        if info and info.get('sweep'):
            return info.get('sweep'), info.get('trial')
        known = catalog().get(name) or {}
        return known.get('sweep'), known.get('trial')

    def summary_sig(directory):
        reports = evaluation_results(directory)
        return (stamp(directory / 'status.json'), stamp(directory / 'result.json'),
                stamp(directory / 'config.json'), stamp(reports[-1]) if reports else None,
                rating_signature(directory))

    def cached_pools(directory, name):
        key = str(directory)
        sig = summary_sig(directory)
        with cache_lock:
            hit = summary_state.get(key)
            if hit and hit[0] == sig:
                return hit[1]
        pools, _ = load_pools()
        rows = pool_summaries(directory, name, pools)
        with cache_lock:
            summary_state[key] = (sig, rows)
            if len(summary_state) > 512:
                summary_state.pop(next(iter(summary_state)))
        return rows

    def pick_pool_summary(rows, primary):
        if not rows:
            return None
        if primary:
            sha8 = primary.get('sha8') or reference_pool.pool_sha8(primary)
            for row in rows:
                if row.get('sha8') == sha8:
                    return row
        return rows[-1]

    def joint_fit():
        with cache_lock:
            return joint_state.get('fit')

    def joint_payload(fit, building=False, error=None):
        pools, primary = load_pools()
        rows = [] if not fit else [dict(row) for row in fit.get('rows') or ()]
        members = [row for row in rows if row.get('kind') in ('scripted', 'random', 'human')]
        return dict(
            building=building, error=error,
            rows=rows, members=members,
            human=None if not fit else fit.get('human'),
            stats=None if not fit else fit.get('stats'),
            issues=[] if not fit else fit.get('issues') or [],
            pools=[pool_brief(item) for item in pools],
            primary=pool_brief(primary),
        )

    def joint_encoded_building():
        return encode_json(joint_payload(joint_fit(), building=True, error=joint_state.get('error')))

    def joint_rebuild():
        pools, _ = load_pools()
        known = catalog()
        sig = joint_ladder.source_signature(root, known, pools, args.human)
        with cache_lock:
            if joint_state['encoded'] is not None and joint_state['sig'] == sig:
                return joint_state['encoded']
            if joint_state['building']:
                wait = True
            else:
                joint_state['building'] = True
                wait = False
        if wait:
            for _ in range(600):
                time.sleep(0.1)
                with cache_lock:
                    if joint_state['sig'] == sig and joint_state['encoded'] is not None and not joint_state['building']:
                        return joint_state['encoded']
            return joint_state['encoded'] or joint_encoded_building()
        try:
            fit = joint_ladder.build(root, known, pools, args.human)
            encoded = encode_json(joint_payload(fit))
            with cache_lock:
                joint_state.update(sig=sig, encoded=encoded, fit=fit, building=False, error=None)
            return encoded
        except Exception as error:
            traceback.print_exc()
            with cache_lock:
                joint_state['building'] = False
                joint_state['error'] = str(error)
            return joint_state['encoded'] or encode_json(joint_payload(None, error=str(error)))

    def joint_current():
        with cache_lock:
            if joint_state['encoded'] is not None:
                return joint_state['encoded']
        return joint_encoded_building()

    def index_row(info, fit):
        directory = info['path']
        status = read_json(directory / 'status.json', {})
        config = read_json(directory / 'config.json', {})
        pool_rows = cached_pools(directory, info['name'])
        joint = joint_for_run(fit, info['name'])
        updated = None
        for path in (directory / 'status.json', directory / 'metrics.jsonl', directory / 'result.json'):
            try:
                value = resolved(path).stat().st_mtime
            except FileNotFoundError:
                continue
            updated = value if updated is None else max(updated, value)
        flags = run_flags(directory)
        return dict(
            name=info['name'], family=info.get('family'), sweep=info.get('sweep'),
            trial=info.get('trial'), rel=info.get('rel'), extra=info.get('extra', False),
            status=status.get('status'), steps=status.get('learner_steps'),
            updated=updated, flags=flags, loadout=loadout_brief(config),
            **config_bits(config),
            pools=[{k: row[k] for k in
                    ('label', 'sha8', 'rating', 'interval', 'unrated', 'game_mismatch') if k in row}
                   for row in pool_rows],
            joint=None if not joint else dict(rating=joint['rating'], interval=joint['interval'],
                                              games=joint.get('games'), steps=joint.get('steps')),
            spark=None if not joint else joint.get('spark') or [],
            unrated=joint is None and all(row.get('unrated', True) for row in pool_rows),
            has_rating=joint is not None or any(not row.get('unrated') for row in pool_rows),
        )

    def index_payload():
        known = catalog()
        fit = joint_fit()
        pools, primary = load_pools()
        sweep_names = set(discover_sweeps(root, extra_runs))
        rows = [index_row(info, fit) for info in known.values()]
        families = {}
        for row in rows:
            key = row['sweep'] or row['family'] or row['name']
            bucket = families.setdefault(key, dict(
                name=key, kind='sweep' if row['sweep'] or key in sweep_names else 'family',
                count=0, rated=0, running=0,
                sweep=row['sweep'] or (key if key in sweep_names else None)))
            bucket['count'] += 1
            if row.get('has_rating'):
                bucket['rated'] += 1
            if row.get('status') == 'running':
                bucket['running'] += 1
        return dict(
            runs=rows,
            families=sorted(families.values(), key=lambda item: (0 if item['kind'] == 'sweep' else 1, item['name'])),
            pools=[pool_brief(item) for item in pools],
            primary=pool_brief(primary),
            joint=None if not fit else dict(stats=fit.get('stats'), building=bool(joint_state.get('building'))),
        )

    def index_current():
        with cache_lock:
            if index_state[1] is not None:
                return index_state[1]
        return index_rebuild()

    def index_rebuild():
        sig = (tuple(sorted(catalog())), joint_state.get('sig'),
               tuple(summary_sig(info['path']) for info in catalog().values()))
        with cache_lock:
            if index_state[1] is not None and index_state[0] == sig:
                return index_state[1]
        encoded = encode_json(index_payload())
        with cache_lock:
            index_state[0], index_state[1] = sig, encoded
        return encoded

    def runs():
        payload = index_payload()
        return payload['runs']

    def run_payload(directory, name):
        status = dict(read_json(directory / 'status.json', {'status': 'starting'}))
        if status.get('status') == 'running':
            try:
                os.kill(status['pid'], 0)
            except (ProcessLookupError, KeyError):
                status['status'] = 'interrupted'
        pools, primary = load_pools()
        pool_rows = cached_pools(directory, name)
        picked = pick_pool_summary(pool_rows, primary)
        reports = evaluation_results(directory)
        evaluations = [{'directory': path.parent.name} for path in reports]
        if reports:
            latest = evaluation_payload(reports[-1])
            if latest:
                evaluations[-1] = latest
        info = catalog().get(name) or {}
        sweep, trial = split_sweep(name, info)
        fit = joint_fit()
        joint = joint_for_run(fit, name)
        flags = (picked or {}).get('flags') or run_flags(directory)
        config = map_catalogue.normalize_record(read_json(directory / 'config.json') or {}, status)
        return dict(
            name=name, config=config, status=status,
            evaluations=evaluations, **loadout_payload(directory),
            ratings=[] if not picked else picked.get('ratings') or [],
            pools=[{k: row[k] for k in row if k != 'ratings'} | dict(
                ratings=row.get('ratings') or []) for row in pool_rows],
            iqm=None if not picked or picked.get('unrated') else dict(
                score=picked['rating'], interval=picked['interval'], last_k=picked.get('last_k'),
                game_mismatch=picked.get('game_mismatch'), label=picked.get('label'),
                sha8=picked.get('sha8')),
            joint=joint,
            human=human_for_run(fit, name),
            flags=flags,
            game_mismatch=bool(picked and picked.get('game_mismatch')),
            unrated=picked is None or picked.get('unrated', True),
            sweep=sweep, trial=trial, family=info.get('family'),
            rate_command=None if not picked else picked.get('rate_command'),
            final_evaluation=latest_report(directory, 'final-evaluations/*/result.json'),
            pool=pool_brief(primary),
            primary=pool_brief(primary),
        )

    def run_alive(directory):
        status = read_json(directory / 'status.json', {})
        if status.get('status') == 'running':
            try:
                os.kill(status['pid'], 0)
                return True
            except (ProcessLookupError, KeyError):
                return False
        return None

    def run_sig(directory):
        reports = evaluation_results(directory)
        return (run_alive(directory), stamp(directory / 'config.json'), stamp(directory / 'status.json'),
                tuple(path.parent.name for path in reports),
                stamp(reports[-1]) if reports else None,
                rating_signature(directory),
                None if not joint_fit() else joint_fit().get('stats'))

    def run_snapshot(directory, name):
        sig = run_sig(directory)
        with cache_lock:
            hit = run_state.pop(name, None)
            if hit and hit[0] == sig:
                run_state[name] = hit
                return hit[1]
        encoded = encode_json(run_payload(directory, name))
        with cache_lock:
            run_state[name] = (sig, encoded)
            while len(run_state) > 16:
                run_state.popitem(last=False)
        return encoded

    def metrics_payload(directory):
        try:
            summary = store.read(directory / 'metrics.jsonl')
        except ValueError:
            raise ValueError('Invalid metrics record')
        records = map_catalogue.run_records(directory)
        summary['rows'] = [map_catalogue.normalize_record(row, *records) for row in summary['rows']]
        matches = map_catalogue.translate_map_matches(summary['map_matches'], *records)
        return dict(metrics=thin_metrics(summary), metrics_total=summary['count'],
                    map_matches=matches, map_catalogue=map_catalogue.IDENTITY)

    def metrics_snapshot(directory, name):
        sig = stamp(directory / 'metrics.jsonl')
        key = name
        with cache_lock:
            hit = metrics_state.pop(key, None)
            if hit and hit[0] == sig:
                metrics_state[key] = hit
                return hit[1]
        encoded = encode_json(metrics_payload(directory))
        with cache_lock:
            metrics_state[key] = (sig, encoded)
            while len(metrics_state) > 4:
                metrics_state.popitem(last=False)
        return encoded

    def eval_snapshot(directory, name, folder):
        result = directory / 'evaluations' / folder / 'result.json'
        if not result.is_file():
            return None
        sig = stamp(result)
        key = (name, folder)
        with cache_lock:
            hit = eval_state.pop(key, None)
            if hit and hit[0] == sig:
                eval_state[key] = hit
                return hit[1]
        payload = evaluation_payload(result)
        if payload is None:
            return None
        encoded = encode_json(payload)
        with cache_lock:
            eval_state[key] = (sig, encoded)
            while len(eval_state) > 16:
                eval_state.popitem(last=False)
        return encoded

    def sweep_payload(name, directory):
        state = read_json(directory / 'sweep.json', {})
        pools, primary = load_pools()
        fit = joint_fit()
        configured = sweep_pool_path(state)
        configured_pool = None
        if configured:
            want = Path(configured)
            if not want.is_absolute():
                want = (HERE.parents[2] / configured).resolve()
            configured_pool = next((item for item in pools if Path(item['path']).resolve() == want), None)
            if configured_pool is None:
                label = joint_ladder.pool_label(want)
                configured_pool = next((item for item in pools if item.get('label') == label), None)
        objective = configured_pool or primary
        objective_sha8 = None if not objective else (objective.get('sha8') or reference_pool.pool_sha8(objective))
        trials = []
        for trial in state.get('trials') or []:
            run_name = f'{name}-{trial["id"]}'
            run_dir = directory / trial['id']
            pool_rows = cached_pools(run_dir, run_name) if run_dir.is_dir() else []
            picked = next((row for row in pool_rows if row.get('sha8') == objective_sha8), None)
            if picked is None:
                picked = pick_pool_summary(pool_rows, objective)
            joint = joint_for_run(fit, run_name)
            point = (trial.get('observations') or [None])[-1]
            trials.append(dict(
                id=trial['id'], status=trial.get('status'), hypers=trial.get('hypers'),
                loadout=loadout_brief(read_json(run_dir / 'config.json', {})),
                flags=trial.get('flags') or (picked or {}).get('flags') or [],
                score=None if not point else point.get('score'),
                cost=None if not point else point.get('cost'),
                steps=None if not point else point.get('steps'),
                rating=None if not picked else picked.get('rating'),
                interval=None if not picked else picked.get('interval'),
                spark=None if not picked else picked.get('spark') or [],
                unrated=True if not picked else picked.get('unrated', True),
                game_mismatch=bool(picked and picked.get('game_mismatch')),
                pools=[{k: row[k] for k in ('label', 'sha8', 'rating', 'interval', 'unrated') if k in row}
                       for row in pool_rows],
                joint=None if not joint else dict(rating=joint['rating'], interval=joint['interval'],
                                                  games=joint.get('games')),
                is_failure=trial.get('is_failure'), early_stop=trial.get('early_stop')))
        return dict(
            name=name, path=str(directory), config=state.get('config'),
            created=state.get('created'), trials=trials,
            pool=pool_brief(objective),
            score_label=None if not objective else f"pool {objective.get('label') or objective_sha8} IQM3",
            pools=[pool_brief(item) for item in pools],
        )

    def sweep_current(name, directory):
        with cache_lock:
            hit = sweep_state.get(name)
            if hit:
                return hit[1]
        return sweep_rebuild(name, directory)

    def sweep_rebuild(name, directory):
        sig = (stamp(directory / 'sweep.json'), joint_state.get('sig'),
               tuple((trial['id'], summary_sig(directory / trial['id']))
                     for trial in (read_json(directory / 'sweep.json', {}).get('trials') or [])
                     if (directory / trial['id']).is_dir()))
        with cache_lock:
            hit = sweep_state.get(name)
            if hit and hit[0] == sig:
                return hit[1]
        encoded = encode_json(sweep_payload(name, directory))
        with cache_lock:
            sweep_state[name] = (sig, encoded)
        return encoded

    def matrix_runs():
        """Runs the loadout matrix compares: the ones whose config declares a selection.

        A run older than the multi-ability work played 2,4,14 and nothing else,
        so it has exactly one row and belongs on the board, not in this matrix.
        """
        found = []
        for name, info in sorted(catalog().items()):
            config = read_json(info['path'] / 'config.json', {}) or {}
            if config.get('loadouts'):
                found.append((name, info, config))
        return found

    def loadout_sig(directory):
        reports = evaluation_results(directory)
        return (stamp(directory / 'config.json'), len(reports),
                tuple((path.parent.name, stamp(path)) for path in reports[-LAST_K:]),
                rating_signature(directory))

    def matrix_payload():
        rows, counts, played = [], {}, {}
        for name, info, config in matrix_runs():
            directory = info['path']
            digests = run_digests(directory)
            if not digests:
                continue
            window = digests[-LAST_K:]
            ratings = loadout_ratings(directory)
            rated = {} if not ratings else {row['key']: row for row in ratings[-1]['rows']}
            cells = {}
            for row in lineup_report.lineup_table(lineup_report.merge_digests(window)['lineups']):
                counts[row['key']] = counts.get(row['key'], 0) + 1
                played[row['key']] = played.get(row['key'], 0) + row['games']
                hit = rated.get(row['key'])
                cells[row['key']] = dict(
                    games=row['games'], timeout_rate=row['timeout_rate'],
                    scripted=(row['families'].get('scripted') or {}).get('win_rate'),
                    random=(row['families'].get('random') or {}).get('win_rate'),
                    rating=None if not hit else hit['rating'])
            status = read_json(directory / 'status.json', {})
            rows.append(dict(name=name, sweep=info.get('sweep'), trial=info.get('trial'),
                             status=status.get('status'), steps=status.get('learner_steps'),
                             loadout=loadout_brief(config), cells=cells, rated=bool(rated),
                             window=len(window), evaluations=len(digests)))
        default = ','.join(str(a) for a in lineup_report.DEFAULT_SELECTION['fixed'])
        keys = sorted(counts, key=lambda key: (key != default, -counts[key], -played[key], key))[:LINEUP_COLUMNS]
        for row in rows:
            row['cells'] = {key: cell for key, cell in row['cells'].items() if key in keys}
        return dict(runs=rows, last_k=LAST_K, default=default, hidden=len(counts) - len(keys),
                    lineups=[dict(lineup_report.lineup_entry(key), runs=counts[key], games=played[key],
                                  comparable=key == default) for key in keys])

    def matrix_rebuild():
        sig = tuple((name, loadout_sig(info['path'])) for name, info, _ in matrix_runs())
        with cache_lock:
            if matrix_state[1] is not None and matrix_state[0] == sig:
                return matrix_state[1]
        encoded = encode_json(matrix_payload())
        with cache_lock:
            matrix_state[0], matrix_state[1] = sig, encoded
        return encoded

    def refresh():
        try:
            catalog(force=True)
            load_pools()
            joint_rebuild()
            index_rebuild()
            for name, directory in discover_sweeps(root, extra_runs).items():
                sweep_rebuild(name, directory)
        except Exception:
            traceback.print_exc()

    def refresh_loop():
        while True:
            time.sleep(2)
            refresh()

    catalog(force=True)
    load_pools()
    threading.Thread(target=refresh, daemon=True).start()
    threading.Thread(target=refresh_loop, daemon=True).start()

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
            encoded = encode_json(data)
            self.send_snapshot(encoded)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/api/ladder':
                return self.send_snapshot(joint_rebuild())
            if path == '/api/index' or path == '/api/runs':
                joint_rebuild()
                return self.send_snapshot(index_rebuild())
            if path == '/api/lineups':
                return self.send_snapshot(matrix_rebuild())
            if path == '/api/sweeps':
                return self.json([{'name': name, 'path': str(directory)}
                                  for name, directory in sorted(discover_sweeps(root, extra_runs).items())])
            if path.startswith('/api/sweep/'):
                name = path[len('/api/sweep/'):]
                directory = discover_sweeps(root, extra_runs).get(name)
                if directory is None:
                    return self.send_error(404)
                joint_rebuild()
                return self.send_snapshot(sweep_rebuild(name, directory))
            if path.startswith('/api/run/'):
                rest = path[len('/api/run/'):]
                name, slash, tail = rest.partition('/')
                directory = run_directory(name)
                if directory is None:
                    return self.send_error(404)
                if tail == 'metrics':
                    try:
                        return self.send_snapshot(metrics_snapshot(directory, name))
                    except ValueError:
                        return self.send_error(500, 'Invalid metrics record')
                if tail.startswith('evaluation/'):
                    folder = tail[len('evaluation/'):]
                    if not re.fullmatch(r'[A-Za-z0-9_.-]+', folder):
                        return self.send_error(404)
                    encoded = eval_snapshot(directory, name, folder)
                    if encoded is None:
                        return self.send_error(404)
                    return self.send_snapshot(encoded)
                if tail:
                    return self.send_error(404)
                joint_rebuild()
                return self.send_snapshot(run_snapshot(directory, name))
            if path.startswith('/runs/'):
                parts = path[len('/runs/'):].split('/')
                directory = run_directory(parts[0]) if parts else None
                if directory is None:
                    return self.send_error(404)
                file = (directory / '/'.join(parts[1:])).resolve()
                if not file.is_relative_to(directory) or file.suffix not in (
                        '.json', '.jsonl', '.rep', '.mp4', '.bin', '.pt', '.jpg', '.gz'):
                    return self.send_error(403)
            elif path in ('/', '/index.html', '/app.js', '/style.css', '/arena.js') or path.startswith(('/run/', '/sweep/', '/ladder', '/lineups')):
                file = HERE / 'web' / ('index.html' if path in ('/', '/index.html') or path.startswith(('/run/', '/sweep/', '/ladder', '/lineups')) else path[1:])
            else:
                return self.send_error(404)
            archived_visual = is_evaluation_visual(file) and not file.is_file() and visuals.exists(file)
            if not file.is_file() and not archived_visual:
                return self.send_error(404)
            if is_evaluation_visual(file) and (archived_visual or not self.headers.get('Range')):
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
                return
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

        def do_HEAD(self):
            return self.do_GET()

        def end_headers(self):
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Bopl quality dashboard: http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
