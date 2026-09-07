"""Saved evaluation matches and on-demand native-input replay preparation."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import quote
import study_data

HERE = Path(__file__).resolve().parent


class ReplayLibrary:
    def __init__(self, root):
        self.python = root / 'venv/bin/python'
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='replay')
        self.lock = threading.RLock()
        self.catalogs = {}
        self.jobs = {}

    def catalog(self, name, directory):
        with self.lock:
            cached = self.catalogs.get((name, str(directory)))
            if cached and time.monotonic() - cached[0] < 15:
                return cached[1]
            matches = []
            evaluated = 0
            for file in sorted((directory / 'evaluations').rglob('result.json')):
                report = study_data.read(file, {})
                for game in report.get('games', []):
                    evaluated += 1
                    if not game.get('replay'):
                        continue
                    replay = (file.parent / game['replay']).resolve()
                    if not replay.is_relative_to(directory.resolve()) or not replay.is_file():
                        continue
                    relative = str(replay.relative_to(directory.resolve()))
                    folder = file.parent.name
                    cross = re.fullmatch(r'crossplay-(\d+)-A(\d+)-B(\d+)', folder)
                    extended = re.fullmatch(r'([AB]-s\d+)-(\d+)-(development|transfer)', folder)
                    if cross:
                        milestone, a, b = cross.groups()
                        run, opponent, suite = f'A-s{a}', f'B-s{b}', 'crossplay'
                    else:
                        run, milestone, suite = extended.groups() if extended else (file.parent.parent.name, '1000000', folder)
                        opponent = game['opponent'].get('spec', {}).get('name', game['opponent']['kind'])
                    players = [run, opponent] if game['side'] == 0 else [opponent, run]
                    win, loss, _ = study_data.outcome(game)
                    matches.append(dict(id=hashlib.sha256(relative.encode()).hexdigest()[:24],
                        name=game['name'], run=run, checkpoint=int(milestone), suite=suite,
                        opponent=opponent, map=game['map'], side=game['side'], seed=game['seed'],
                        outcome='win' if win else 'loss' if loss else 'draw', players=players,
                        seconds=game['ticks']/60, replay_url=f'/studies/{quote(name)}/{quote(relative)}',
                        replay_sha256=game['replay_sha256'], relative=relative,
                        final_players=game['final_players'], ticks=game['ticks']))
            matches.sort(key=lambda m: (-m['checkpoint'], m['suite'] != 'crossplay', m['run'], m['opponent'], m['map'], m['seed'], m['side']))
            value = dict(matches=matches, saved=len(matches), evaluated=evaluated)
            self.catalogs[(name, str(directory))] = (time.monotonic(), value)
            return value

    def replay(self, name, directory, match_id, prepare=False):
        with self.lock:
            match = next((m for m in self.catalog(name, directory)['matches'] if m['id'] == match_id), None)
            if match is None:
                raise KeyError(match_id)
            key = (str(directory), match['replay_sha256'])
            output = directory / 'replay-cache' / (match['replay_sha256'] + '.json.gz')
            if output.is_file():
                return dict(status='ready', url=f'/studies/{quote(name)}/replay-cache/{output.name}')
            job = self.jobs.get(key)
            if job:
                return dict(job)
            if not prepare:
                return dict(status='not prepared')
            if sum(j['status'] in ('queued', 'preparing') for j in self.jobs.values()) >= 4:
                return dict(status='busy', error='Four replays are being prepared. Try again shortly.')
            job = self.jobs[key] = dict(status='queued')
            self.pool.submit(self._prepare, directory, match, output, job)
            return dict(job)

    def _prepare(self, directory, match, output, job):
        try:
            with self.lock:
                job['status'] = 'preparing'
            replay = directory / match['relative']
            if hashlib.sha256(replay.read_bytes()).hexdigest() != match['replay_sha256']:
                raise ValueError('Saved replay checksum changed')
            output.parent.mkdir(parents=True, exist_ok=True)
            # Publish only after checking the final game state against evaluation.
            pending = output.with_name(output.name + '.pending')
            pending.unlink(missing_ok=True)
            with output.with_suffix('.prepare.log').open('w') as log:
                subprocess.run([str(self.python), str(HERE/'prepare_replay.py'), '--replay', str(replay), '--output', str(pending)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180, start_new_session=True)
            with gzip.open(pending, 'rt') as stream:
                value = json.load(stream)
            if value['frames'][-1]['players'] != match['final_players'] or value['frames'][-1]['t'] != match['ticks']:
                raise ValueError('Replay final state differs from the recorded evaluation')
            pending.replace(output)
            with self.lock:
                job['status'] = 'ready'
        except Exception as error:
            with self.lock:
                job.update(status='error', error=str(error))
