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
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from metrics_store import store

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'sweep')]
import reference_pool


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


def run_flags(directory):
    result = read_json(directory / 'result.json', {})
    if result.get('flags') is not None:
        return result['flags']
    reports = sorted(directory.glob('evaluations/*/result.json'), key=step_order)
    if reports:
        return read_json(reports[-1], {}).get('flags') or []
    return []


def run_curve(directory, pool):
    rows = []
    for payload in reference_pool.rated_curve(directory, pool):
        rows.append(dict(checkpoint=payload['checkpoint'], learner_steps=payload['learner_steps'],
                         rating=payload['rating'], interval=payload['interval'],
                         games=payload.get('games'), source=payload.get('source'),
                         game_mismatch=payload.get('game_mismatch', False)))
    return rows


def run_summary(directory, pool, name, last_k=3):
    curve = run_curve(directory, pool)
    summary = reference_pool.run_iqm(directory, pool, last_k=last_k) if curve else None
    return dict(
        name=name, kind='run', path=str(directory),
        rating=None if not summary else summary['score'],
        interval=None if not summary else summary['interval'],
        spark=[{'steps': r['learner_steps'], 'rating': r['rating'],
                'lo': r['interval'][0], 'hi': r['interval'][1]} for r in curve],
        flags=run_flags(directory),
        game_mismatch=bool(summary and summary['game_mismatch']),
        unrated=summary is None,
        rate_command=None if summary else reference_pool.rate_command(directory, pool.get('path')),
        last_k=None if not summary else [
            dict(checkpoint=r['checkpoint'], learner_steps=r['learner_steps'],
                 rating=r['rating'], interval=r['interval'], games=r.get('games'))
            for r in summary['last_k']],
    )


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

    def runs():
        candidates = []
        named = named_runs(extra_runs, run_globs)
        for file in [*root.glob('*/config.json'), *(path / 'config.json' for path in named.values())]:
            try:
                config = read_json(file, {})
            except (OSError, ValueError):
                continue
            if not (config and config.get('schema') in (1, 2) and config.get('algorithm') == 'recurrent PPO'):
                continue
            name = next((n for n, p in named.items() if p == file.parent), file.parent.name)
            candidates.append(dict(name=name, created=file.stat().st_mtime,
                                   workspace=str(file.parent) if name in named else None,
                                   status=read_json(file.parent / 'status.json', {})))
        return sorted(candidates, key=lambda r: r['created'], reverse=True)

    def run_directory(name):
        named = named_runs(extra_runs, run_globs)
        if name in named:
            return named[name]
        if re.fullmatch(r'[A-Za-z0-9_.-]+', name) and (root / name / 'config.json').is_file():
            return root / name
        return None

    def split_sweep(name):
        for prefix in sorted(run_globs, key=len, reverse=True):
            token = prefix + '-'
            if name.startswith(token):
                return prefix, name[len(token):]
        return None, None

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
        human = None
        if pool and args.human.is_dir() and (args.human / 'games.jsonl').is_file():
            try:
                human = reference_pool.rate_human(pool, args.human)
                members.append(dict(
                    name='human', kind='human', member_kind='human',
                    rating=human['rating'], interval=human['interval'],
                    games=human['games'], spark=[], flags=[],
                    game_mismatch=True, unrated=False, sweep=None, trial=None,
                    caveats=human.get('caveats'), records=human.get('records')))
            except ValueError:
                human = None
        rows = []
        for entry in runs():
            directory = run_directory(entry['name'])
            if directory is None or pool is None:
                continue
            row = run_summary(directory, pool, entry['name'])
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

    def run_payload(directory, name):
        status = dict(read_json(directory / 'status.json', {'status': 'starting'}))
        if status.get('status') == 'running':
            try:
                os.kill(status['pid'], 0)
            except (ProcessLookupError, KeyError):
                status['status'] = 'interrupted'
        try:
            summary = store.read(directory / 'metrics.jsonl')
        except ValueError:
            raise ValueError('Invalid metrics record')
        evaluations = []
        for file in sorted(directory.glob('evaluations/*/result.json'), key=step_order):
            report = compact_report(file)
            if not report:
                continue
            report['directory'] = file.parent.name
            for game in report['games']:
                video = file.parent / (game['name'] + '.mp4')
                if video.exists():
                    game['video'] = video.name
                poster = video.with_suffix('.jpg')
                if poster.exists():
                    game['poster'] = poster.name
                mind = file.parent / (game['name'] + '.mind.json.gz')
                if mind.exists():
                    game['mind'] = mind.name
            evaluations.append(report)
        ratings = run_curve(directory, pool) if pool else []
        iqm = reference_pool.run_iqm(directory, pool) if pool and ratings else None
        sweep, trial = split_sweep(name)
        return dict(
            name=name, config=read_json(directory / 'config.json'), status=status,
            metrics=summary['rows'], metrics_total=summary['count'],
            map_matches=summary['map_matches'], evaluations=evaluations,
            ratings=ratings, iqm=iqm, flags=run_flags(directory),
            game_mismatch=bool(iqm and iqm['game_mismatch']),
            unrated=not ratings, sweep=sweep, trial=trial,
            rate_command=None if ratings else reference_pool.rate_command(directory, pool['path'] if pool else None),
            final_evaluation=latest_report(directory, 'final-evaluations/*/result.json'),
            pool=None if not pool else dict(sha8=reference_pool.pool_sha8(pool),
                                            sha256=pool['pool_sha256']))

    def sweep_payload(name, directory):
        state = read_json(directory / 'sweep.json', {})
        trials = []
        for trial in state.get('trials') or []:
            run_dir = directory / trial['id']
            summary = run_summary(run_dir, pool, f'{name}-{trial["id"]}') if pool and run_dir.is_dir() else {}
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

    responses = OrderedDict()
    response_lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, fmt, *arguments):
            if len(arguments) < 2 or str(arguments[1]) not in ('200', '206', '304'):
                super().log_message(fmt, *arguments)

        def json(self, data):
            payload = json.dumps(data, allow_nan=False, separators=(',', ':')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            if len(payload) > 16384 and 'gzip' in self.headers.get('Accept-Encoding', ''):
                payload = gzip.compress(payload, 1)
                self.send_header('Content-Encoding', 'gzip')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/api/ladder':
                return self.json(ladder_payload())
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
                return self.json(sweep_payload(name, directory))
            if path.startswith('/api/run/'):
                name = path[len('/api/run/'):]
                directory = run_directory(name)
                if directory is None:
                    return self.send_error(404)
                try:
                    with response_lock:
                        dependencies = [directory / file for file in
                                        ('config.json', 'status.json', 'metrics.jsonl')]
                        for result in sorted(directory.glob('evaluations/*/result.json')):
                            dependencies.extend((result, result.parent))
                        dependencies.append(reference_pool.rating_dir(directory, pool) if pool else directory)
                        status = read_json(directory / 'status.json', {})
                        alive = True
                        if status.get('status') == 'running':
                            try:
                                os.kill(status['pid'], 0)
                            except (ProcessLookupError, KeyError):
                                alive = False
                        signature = (alive, tuple((str(p), stamp(p)) for p in dependencies))
                        cached = responses.pop(name, None)
                        if cached is None or cached[0] != signature:
                            payload = json.dumps(run_payload(directory, name),
                                                 allow_nan=False, separators=(',', ':')).encode()
                            cached = (signature, '"' + hashlib.sha256(payload).hexdigest() + '"',
                                      payload, gzip.compress(payload, 1))
                        responses[name] = cached
                        while len(responses) > 4:
                            responses.popitem(last=False)
                except ValueError:
                    return self.send_error(500, 'Invalid metrics record')
                _, etag, payload, compressed = cached
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(304)
                    self.send_header('ETag', etag)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('ETag', etag)
                self.send_header('Vary', 'Accept-Encoding')
                if 'gzip' in self.headers.get('Accept-Encoding', ''):
                    payload = compressed
                    self.send_header('Content-Encoding', 'gzip')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                if self.command != 'HEAD':
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                return
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
