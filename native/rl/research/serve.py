"""Quality dashboard: one pool-v5 Elo scale, three views (ladder / run / sweep)."""
import argparse
from collections import OrderedDict
import glob
import gzip
import hashlib
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

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'sweep')]
import reference_pool

CHART_KEYS = (
    'step', 'learner_steps', 'elapsed', 'sps', 'learner_sps', 'policy_loss',
    'value_loss', 'entropy', 'kl', 'clip_fraction', 'gradient_norm',
    'explained_variance', 'mean_reward', 'entropy_coefficient', 'learning_rate',
    'advantage_std', 'entropy_aim', 'entropy_buttons', 'cuda_memory_mb',
)
CHART_POINTS = 360
LAST_K = 3


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError, OSError):
        return default


def stamp(path):
    try:
        value = Path(path).stat()
    except FileNotFoundError:
        return None
    return value.st_ino, value.st_mtime_ns, value.st_size


def encode_json(data):
    raw = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
    return raw, gzip.compress(raw, 1), '"' + hashlib.sha256(raw).hexdigest() + '"'


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


def named_runs(extra_runs, run_globs):
    found = dict(extra_runs)
    for prefix, pattern in run_globs.items():
        for match in sorted(glob.glob(pattern)):
            directory = Path(match)
            if directory.is_dir() and re.fullmatch(r'[A-Za-z0-9_.-]+', directory.name):
                found.setdefault(f'{prefix}-{directory.name}', directory.resolve())
    return found


def discover_sweeps(root, extra_runs, run_globs):
    found = {}
    for prefix, pattern in run_globs.items():
        for match in glob.glob(pattern):
            parent = Path(match).resolve().parent
            if (parent / 'sweep.json').is_file():
                found[prefix] = parent
                break
    for file in root.glob('sweeps/*/sweep.json'):
        found.setdefault(file.parent.name, file.parent)
    return found


def evaluation_results(directory):
    root = Path(directory) / 'evaluations'
    if not root.is_dir():
        return []
    found = [child / 'result.json' for child in root.iterdir() if (child / 'result.json').is_file()]
    found.sort(key=step_order)
    return found


def rating_signature(directory, pool):
    if not pool:
        return None
    files = []
    for folder in (Path(directory) / 'reference', reference_pool.rating_dir(directory, pool)):
        if not folder.is_dir():
            continue
        for path in folder.glob('*.json'):
            files.append((path.name, stamp(path)))
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
    parser.add_argument('--run-root-glob', action='append', default=[], metavar='NAME=GLOB',
                        help='Register every run directory matching GLOB as NAME-<dirname>')
    parser.add_argument('--pool', type=Path, default=reference_pool.DEFAULT_POOL,
                        help='Current reference pool.json (ladder scale)')
    parser.add_argument('--human', type=Path, default=reference_pool.DEFAULT_HUMAN,
                        help='human/mac directory (games.jsonl + excluded.json)')
    args = parser.parse_args()
    root = args.root.resolve()
    extra_runs, run_globs = {}, {}
    for value in args.run_root:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('run_root requires NAME=PATH')
        extra_runs[name] = Path(path).resolve()
    for value in args.run_root_glob:
        name, separator, path = value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('run_root_glob requires NAME=GLOB')
        run_globs[name] = path

    pool = None
    if args.pool and args.pool.is_file():
        pool = reference_pool.load_pool(args.pool)

    cache_lock = threading.Lock()
    named_state = [None, None]
    summary_state = {}
    human_state = [None, None]
    ladder_state = [None, None]
    sweep_state = {}
    run_state = OrderedDict()
    eval_state = OrderedDict()
    metrics_state = OrderedDict()

    def named():
        sig = (tuple((name, str(path), stamp(path)) for name, path in extra_runs.items()),
               tuple((name, pattern, stamp(Path(pattern).parent)) for name, pattern in run_globs.items()))
        with cache_lock:
            if named_state[0] == sig:
                return named_state[1]
        found = named_runs(extra_runs, run_globs)
        with cache_lock:
            named_state[0], named_state[1] = sig, found
        return found

    def runs():
        candidates = []
        known = named()
        for file in [*root.glob('*/config.json'), *(path / 'config.json' for path in known.values())]:
            try:
                config = read_json(file, {})
            except (OSError, ValueError):
                continue
            if not (config and config.get('schema') in (1, 2) and config.get('algorithm') == 'recurrent PPO'):
                continue
            name = next((n for n, p in known.items() if p == file.parent), file.parent.name)
            candidates.append(dict(name=name, created=file.stat().st_mtime,
                                   workspace=str(file.parent) if name in known else None,
                                   status=read_json(file.parent / 'status.json', {})))
        return sorted(candidates, key=lambda r: r['created'], reverse=True)

    def run_directory(name):
        known = named()
        if name in known:
            return known[name]
        if re.fullmatch(r'[A-Za-z0-9_.-]+', name) and (root / name / 'config.json').is_file():
            return root / name
        return None

    def split_sweep(name):
        for prefix in sorted(run_globs, key=len, reverse=True):
            token = prefix + '-'
            if name.startswith(token):
                return prefix, name[len(token):]
        return None, None

    def human_rating():
        if not pool or not args.human.is_dir() or not (args.human / 'games.jsonl').is_file():
            return None
        sig = (stamp(args.human / 'games.jsonl'), stamp(args.human / 'excluded.json'), pool['pool_sha256'])
        with cache_lock:
            if human_state[0] == sig:
                return human_state[1]
        try:
            value = reference_pool.rate_human(pool, args.human)
        except ValueError:
            value = None
        with cache_lock:
            human_state[0], human_state[1] = sig, value
        return value

    def summary_sig(directory):
        reports = evaluation_results(directory)
        return (stamp(directory / 'status.json'), stamp(directory / 'result.json'),
                stamp(reports[-1]) if reports else None, rating_signature(directory, pool))

    def cached_summary(directory, name):
        key = str(directory)
        sig = summary_sig(directory)
        with cache_lock:
            hit = summary_state.get(key)
            if hit and hit[0] == sig:
                return hit[1]
        row = run_summary(directory, pool, name)
        with cache_lock:
            summary_state[key] = (sig, row)
            if len(summary_state) > 256:
                summary_state.pop(next(iter(summary_state)))
        return row

    def ladder_payload():
        members = []
        if pool:
            for member in pool['members']:
                members.append(dict(
                    name=member['name'], kind='pool', member_kind=member['kind'],
                    rating=member['rating'], interval=member.get('interval'),
                    games=member.get('games'), wins=member.get('wins'),
                    losses=member.get('losses'), draws=member.get('draws'),
                    spark=[], flags=[], game_mismatch=False, unrated=False,
                    sweep=None, trial=None))
        human = human_rating()
        if human:
            members.append(dict(
                name='human', kind='human', member_kind='human',
                rating=human['rating'], interval=human['interval'],
                games=human['games'], spark=[], flags=[],
                game_mismatch=True, unrated=False, sweep=None, trial=None,
                caveats=human.get('caveats'), records=human.get('records')))
        rows = []
        for entry in runs():
            directory = run_directory(entry['name'])
            if directory is None:
                continue
            row = dict(cached_summary(directory, entry['name']))
            row.pop('ratings', None)
            sweep, trial = split_sweep(entry['name'])
            row.update(sweep=sweep, trial=trial, kind='sweep-trial' if sweep else 'run',
                       status=(entry.get('status') or {}).get('status'),
                       created=entry.get('created'))
            rows.append(row)
        return dict(
            pool=None if not pool else dict(
                path=pool['path'], sha256=pool['pool_sha256'], sha8=reference_pool.pool_sha8(pool),
                sim_sha256=(pool.get('game') or {}).get('sha256'),
                maps=pool.get('maps'), members=pool['members']),
            human=None if not human else dict(rating=human['rating'], interval=human['interval'],
                                              games=human['games'], caveats=human.get('caveats'),
                                              records=human.get('records')),
            rows=rows, members=members,
        )

    def ladder_sig():
        parts = [stamp(args.pool), stamp(args.human / 'games.jsonl'), stamp(args.human / 'excluded.json')]
        for entry in runs():
            directory = run_directory(entry['name'])
            parts.append((entry['name'], None if directory is None else summary_sig(directory)))
        return tuple(parts)

    def ladder_current():
        with cache_lock:
            if ladder_state[1] is not None:
                return ladder_state[1]
        return ladder_rebuild()

    def ladder_rebuild():
        sig = ladder_sig()
        with cache_lock:
            if ladder_state[1] is not None and ladder_state[0] == sig:
                return ladder_state[1]
        encoded = encode_json(ladder_payload())
        with cache_lock:
            ladder_state[0], ladder_state[1] = sig, encoded
        return encoded

    def run_payload(directory, name):
        status = dict(read_json(directory / 'status.json', {'status': 'starting'}))
        if status.get('status') == 'running':
            try:
                os.kill(status['pid'], 0)
            except (ProcessLookupError, KeyError):
                status['status'] = 'interrupted'
        summary = cached_summary(directory, name)
        reports = evaluation_results(directory)
        evaluations = [{'directory': path.parent.name} for path in reports]
        if reports:
            latest = evaluation_payload(reports[-1])
            if latest:
                evaluations[-1] = latest
        sweep, trial = split_sweep(name)
        return dict(
            name=name, config=read_json(directory / 'config.json'), status=status,
            evaluations=evaluations, ratings=summary.get('ratings') or [],
            iqm=None if summary['unrated'] else dict(
                score=summary['rating'], interval=summary['interval'], last_k=summary['last_k'],
                game_mismatch=summary['game_mismatch']),
            flags=summary.get('flags') or [],
            game_mismatch=summary.get('game_mismatch', False),
            unrated=summary.get('unrated', True), sweep=sweep, trial=trial,
            rate_command=summary.get('rate_command'),
            final_evaluation=latest_report(directory, 'final-evaluations/*/result.json'),
            pool=None if not pool else dict(sha8=reference_pool.pool_sha8(pool),
                                            sha256=pool['pool_sha256']))

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
                rating_signature(directory, pool))

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
            while len(run_state) > 8:
                run_state.popitem(last=False)
        return encoded

    def metrics_payload(directory):
        try:
            summary = store.read(directory / 'metrics.jsonl')
        except ValueError:
            raise ValueError('Invalid metrics record')
        return dict(metrics=thin_metrics(summary), metrics_total=summary['count'],
                    map_matches=summary['map_matches'])

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
        trials = []
        for trial in state.get('trials') or []:
            run_dir = directory / trial['id']
            summary = cached_summary(run_dir, f'{name}-{trial["id"]}') if pool and run_dir.is_dir() else {}
            point = (trial.get('observations') or [None])[-1]
            trials.append(dict(
                id=trial['id'], status=trial.get('status'), hypers=trial.get('hypers'),
                flags=trial.get('flags') or summary.get('flags') or [],
                score=None if not point else point.get('score'),
                cost=None if not point else point.get('cost'),
                steps=None if not point else point.get('steps'),
                rating=summary.get('rating'), interval=summary.get('interval'),
                spark=summary.get('spark') or [], unrated=summary.get('unrated', True),
                game_mismatch=summary.get('game_mismatch', False),
                is_failure=trial.get('is_failure'), early_stop=trial.get('early_stop')))
        return dict(name=name, path=str(directory), config=state.get('config'),
                    created=state.get('created'), trials=trials,
                    pool=None if not pool else dict(sha8=reference_pool.pool_sha8(pool),
                                                    sha256=pool['pool_sha256']))

    def sweep_current(name, directory):
        with cache_lock:
            hit = sweep_state.get(name)
            if hit:
                return hit[1]
        return sweep_rebuild(name, directory)

    def sweep_rebuild(name, directory):
        sig = (stamp(directory / 'sweep.json'),
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

    def refresh():
        try:
            ladder_rebuild()
            for name, directory in discover_sweeps(root, extra_runs, run_globs).items():
                sweep_rebuild(name, directory)
        except Exception:
            traceback.print_exc()

    def refresh_loop():
        while True:
            time.sleep(2)
            refresh()

    refresh()
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
                return self.send_snapshot(ladder_current())
            if path == '/api/runs':
                return self.json(runs())
            if path == '/api/sweeps':
                return self.json([{'name': name, 'path': str(directory)}
                                  for name, directory in sorted(discover_sweeps(root, extra_runs, run_globs).items())])
            if path.startswith('/api/sweep/'):
                name = path[len('/api/sweep/'):]
                directory = discover_sweeps(root, extra_runs, run_globs).get(name)
                if directory is None:
                    return self.send_error(404)
                return self.send_snapshot(sweep_current(name, directory))
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
            elif path in ('/', '/index.html', '/app.js', '/style.css', '/arena.js') or path.startswith(('/run/', '/sweep/')):
                file = HERE / 'web' / ('index.html' if path in ('/', '/index.html') or path.startswith(('/run/', '/sweep/')) else path[1:])
            else:
                return self.send_error(404)
            if not file.is_file():
                return self.send_error(404)
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
