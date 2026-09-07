"""Read Fable's experiment log; serve a live chart or perform one hourly check."""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import math
from pathlib import Path
import subprocess
import time
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent


def read_results(path):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return [], []
    # Ignore an in-progress final write without hiding malformed completed rows.
    if text and not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] + '\n' if '\n' in text else ''
    rows, errors = [], []
    for line, row in enumerate(csv.DictReader(io.StringIO(text), delimiter='\t'), 2):
        try:
            timestamp = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError('timestamp needs timezone')
            score = float(row['learner_dps_wall']) if row['learner_dps_wall'] else None
            if score is not None and (not math.isfinite(score) or score <= 0):
                raise ValueError('score must be positive and finite')
            if row['status'] not in ('baseline', 'keep', 'discard', 'invalid'):
                raise ValueError('unknown status')
            if not row['workload'] or not row['experiment']:
                raise ValueError('missing workload or experiment')
            if row['status'] != 'invalid' and score is None:
                raise ValueError('missing score')
            rows.append({**{k: row.get(k, '') for k in (
                'commit', 'experiment', 'workload', 'status', 'description')},
                'timestamp_utc': timestamp.astimezone(timezone.utc).isoformat(),
                'learner_dps_wall': score})
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f'Row {line}: {error}')
    return sorted(rows, key=lambda row: row['timestamp_utc']), errors


def process_status(config):
    try:
        result = subprocess.run(['claude', 'agents', '--json', '--cwd', config['research_root']],
                                capture_output=True, text=True, timeout=15, check=True)
        sessions = json.loads(result.stdout)
        session = next((s for s in sessions if s['id'] == config['session_id']), None)
        if session is None:
            return {'state': 'not running', 'settings_verified': False}
        args = Path(f"/proc/{session['pid']}/cmdline").read_bytes().decode().split('\0')
        def option(name):
            return args[args.index(name) + 1] if name in args else None
        return {'state': session.get('status', session.get('state', 'unknown')),
                'pid': session['pid'], 'model': option('--model'), 'effort': option('--effort'),
                'full_access': '--dangerously-skip-permissions' in args,
                'settings_verified': option('--model') == 'claude-fable-5-1'
                and option('--effort') == 'high' and '--dangerously-skip-permissions' in args}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {'state': 'check failed', 'settings_verified': False, 'error': str(error)}


@lru_cache(maxsize=8)
def chart(serialized, extension):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import dates as mdates
    from matplotlib.figure import Figure
    rows = json.loads(serialized)
    groups = {}
    for row in rows:
        groups.setdefault(row['workload'], []).append(row)
    fig = Figure(figsize=(10.8, max(3.8, 3.2 * len(groups))), facecolor='#f7f5ed', layout='constrained')
    if not groups:
        ax = fig.subplots()
        ax.set_axis_off()
        ax.text(.5, .57, 'Establishing the baseline', ha='center', va='center', fontsize=23,
                color='#244f3f', transform=ax.transAxes)
        ax.text(.5, .40, 'Measured experiments will appear here automatically.\nNo speedup claimed yet.',
                ha='center', va='center', fontsize=13, color='#66685e', transform=ax.transAxes)
    for index, (workload, values) in enumerate(groups.items(), 1):
        ax = fig.add_subplot(len(groups), 1, index, facecolor='#f7f5ed')
        points = [r for r in values if r['learner_dps_wall'] is not None and r['status'] != 'invalid']
        best, times, scores = 0, [], []
        for row in points:
            stamp = datetime.fromisoformat(row['timestamp_utc'])
            score = row['learner_dps_wall']
            retained = row['status'] in ('baseline', 'keep')
            ax.scatter(stamp, score, color='#286248' if retained else '#b58243',
                       s=45, zorder=3, marker='o' if retained else 'x')
            if retained:
                best = max(best, score)
            if best:
                times.append(stamp)
                scores.append(best)
        if times:
            ax.step(times, scores, where='post', color='#286248', linewidth=2, label='Best retained')
            ax.legend(frameon=False, loc='upper left')
        ax.set_title(workload, loc='left', fontsize=13, color='#244f3f')
        ax.set_ylabel('Learner decisions / second')
        ax.set_xlabel('Experiment completed · UTC')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d %b', tz=timezone.utc))
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    output = io.BytesIO()
    fig.savefig(output, format=extension, dpi=140, metadata={'Creator': 'Bopl research monitor'}
                if extension == 'svg' else None)
    return output.getvalue()


def snapshot(config, output, check=False):
    rows, errors = read_results(Path(config['research_root']) / 'native/rl/autoresearch/results.tsv')
    report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'experiments': rows,
              'errors': errors, 'session_id': config['session_id'], 'session': process_status(config),
              'graph_url': config['graph_url'], 'last_hourly_check': None}
    hourly = output / 'hourly-check.json'
    if hourly.exists():
        report['last_hourly_check'] = json.loads(hourly.read_text()).get('updated_at')
    output.mkdir(parents=True, exist_ok=True)
    if check:
        report['last_hourly_check'] = report['updated_at']
        temporary = hourly.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(hourly)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8877)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.config.parent
    if args.check:
        report = snapshot(config, output, check=True)
        serialized = json.dumps(report['experiments'])
        for extension in ('svg', 'png'):
            (output / f'progress.{extension}').write_bytes(chart(serialized, extension))
        print(json.dumps(report))
        return
    cache = {'at': 0, 'report': None}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            try:
                name = urlparse(self.path).path.rsplit('/', 1)[-1]
                if name in ('', 'index.html', 'autoresearch'):
                    payload, mime = (HERE / 'progress.html').read_bytes(), 'text/html; charset=utf-8'
                elif name == 'health':
                    payload, mime = b'{"ok":true}', 'application/json'
                elif name in ('data.json', 'progress.svg', 'progress.png'):
                    if time.monotonic() - cache['at'] > 10:
                        cache.update(at=time.monotonic(), report=snapshot(config, output))
                    report = cache['report']
                    if name == 'data.json':
                        payload, mime = json.dumps(report).encode(), 'application/json'
                    else:
                        extension = name.split('.')[-1]
                        payload = chart(json.dumps(report['experiments']), extension)
                        mime = 'image/svg+xml' if extension == 'svg' else 'image/png'
                else:
                    return self.send_error(404)
                self.send_response(200)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            except (OSError, ValueError, KeyError) as error:
                self.send_error(503, 'Monitor data unavailable')
                print(f'monitor error: {error}', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
