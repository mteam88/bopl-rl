"""Bounded-memory, full-state actor comparison; one JSON record per frame.

A case ends only with a validated result record. Hashes bind the exact input,
Unity assembly and output; neither truncated streams nor extra frames are accepted.
"""
from collections import Counter
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from reference import ROOT, first_difference

SCHEMA = 'bopl-actor-stream-v1'


def encode(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True) + '\n'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records(stream):
    for line in stream:
        yield json.loads(line)


class Protocol:
    def __init__(self, queries):
        self.queries = queries
        self.case = self.action = 0
        self.counts = Counter()
        self.players = Counter()
        self.phases = Counter()
        self.events = Counter()
        self.errors = Counter()
        self.invisibility = Counter()
        self.player_states = Counter()
        self.error_cases = []
        self.previous = None

    def frame(self, value, query):
        required = {'actors', 'scene', 'report', 'ticks', 'time', 'level_time', 'inputs'}
        if not isinstance(value, dict) or not required <= set(value):
            raise ValueError('Incomplete actor state')
        if len(value['actors']) != len(query['runtime']['actors']) or len(value['inputs']) != len(query['runtime']['scheduler']['inputs']):
            raise ValueError('Incorrect actor or input count')
        if not {'phases', 'events', 'pushes', 'physics'} <= set(value['report']):
            raise ValueError('Incomplete actor phase report')
        if query.get('player_state') and len(value.get('players', [])) != len(query['runtime']['scheduler']['players']):
            raise ValueError('Incomplete shared player state')

    def accept(self, row):
        if self.case >= len(self.queries) or type(row.get('case')) is not int or row['case'] != self.case:
            raise ValueError('Unexpected actor stream case')
        query = self.queries[self.case]
        if set(row) == {'case', 'action', 'state'}:
            if type(row['action']) is not int or row['action'] != self.action or self.action >= len(query['actions']):
                raise ValueError('Missing, duplicate or excess actor frame')
            frame = row['state']
            self.frame(frame, query)
            self.counts['ticks'] += 1
            self.counts['shoves'] += len(frame['report']['pushes'])
            self.phases.update(p['kind'] for p in frame['report']['phases'])
            self.events.update(e['event']['kind'] for e in frame['report']['events'])
            physics = frame['report']['physics']
            if physics is not None:
                self.counts['physics_stages'] += 1
                self.counts['collision_callbacks'] += len(physics['callbacks'])
            if 'players' in frame:
                if self.previous is None:
                    self.previous = [p['invisible'] for p in query['runtime']['scheduler']['players']]
                if len(frame['players']) != len(self.previous):
                    raise ValueError('Incorrect shared player count')
                for i, player in enumerate(frame['players']):
                    invisible = player['invisible']
                    self.player_states['invisible' if invisible else 'visible'] += 1
                    self.invisibility['became_invisible'] += not self.previous[i] and invisible
                    self.invisibility['expired'] += self.previous[i] and not invisible
                    self.previous[i] = invisible
            for event in frame['report']['events']:
                self.invisibility['cast_audio_requests'] += event['event'] == {'kind': 'sound', 'value': 'invisibility'}
            for actor in frame['actors']:
                for ability in actor.get('slime', {}).get('abilities', {}).get('pool', []):
                    handler = ability.get('handler')
                    if handler and handler['kind'] == 'invisibility':
                        self.invisibility['sentinel_cooldown_snapshots'] += ability['cooldown'] == 10000 * (1 << 32)
                        self.invisibility['restored_cooldown_snapshots'] += ability['cooldown'] == handler['restore_cooldown']
            self.action += 1
        elif set(row) == {'case', 'result'}:
            result = row['result']
            if type(result.get('completed')) is not int or result['completed'] != self.action:
                raise ValueError('Incorrect actor completion count')
            if 'error' in result:
                if set(result) != {'error', 'action', 'completed', 'failed'} or result['action'] != self.action or self.action >= len(query['actions']):
                    raise ValueError('Incomplete actor failure state')
                if not isinstance(result['error'], str):
                    raise ValueError('Invalid actor error')
                self.frame(result['failed'], query)
                self.errors[result['error']] += 1
                self.error_cases.append(self.case)
            elif set(result) != {'completed'} or self.action != len(query['actions']):
                raise ValueError('Incomplete actor episode')
            else:
                self.counts['complete_episodes'] += 1
            self.players[len(query['runtime']['scheduler']['players'])] += 1
            self.case += 1
            self.action = 0
            self.previous = None
        else:
            raise ValueError('Unknown actor stream record')

    def finish(self):
        if self.case != len(self.queries) or self.action:
            raise ValueError('Actor stream has no complete end record')
        return dict(cases=self.case, **self.counts, players=dict(self.players),
                    phases=dict(self.phases), events=dict(self.events), errors=dict(self.errors),
                    error_cases=self.error_cases, invisibility=dict(self.invisibility),
                    player_states=dict(self.player_states))


def capture(directory, queries, reuse=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    queries = [dict(q, stream_frames=True) for q in queries]
    source, output = directory / 'input.jsonl', directory / 'unity.jsonl'
    provenance = directory / 'reference-sha256.json'
    if reuse:
        with source.open() as stream:
            if list(records(stream)) != queries:
                raise ValueError('Cannot reuse changed actor queries')
    else:
        if any(p.exists() for p in (source, output, provenance)):
            raise FileExistsError('Use a fresh directory for actor stream evidence')
        with source.open('x') as stream:
            for query in queries:
                stream.write(encode(query))
        assembly = ROOT / 'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll'
        assembly_hash = digest(assembly)
        env = dict(os.environ, BOPL_KERNEL_INPUT=str(source), BOPL_KERNEL_OUTPUT=str(output))
        for key in ('BOPL_TRACE', 'BOPL_TRACE_TICKS', 'BOPL_TRACE_BATCH'):
            env.pop(key, None)
        log = directory / 'unity.log'
        with (directory / 'stdout.log').open('w') as stdout:
            subprocess.run([str(ROOT / 'Builds/Linux/BoplBattle.x86_64'),
                            str(ROOT / 'native/fixtures/movement.rep'), '-batchmode', '-nographics',
                            '-logFile', str(log)], env=env, stdout=stdout,
                           stderr=subprocess.STDOUT, check=True, timeout=1800)
        if 'REFERENCE_KERNEL_COMPLETE' not in log.read_text() or digest(assembly) != assembly_hash:
            raise ValueError('Incomplete capture or changed Unity assembly')
        protocol = Protocol(queries)
        with output.open() as stream:
            for row in records(stream):
                protocol.accept(row)
        protocol.finish()
        provenance.write_text(encode(dict(assembly=assembly_hash, input=digest(source), output=digest(output))))
    hashes = json.loads(provenance.read_text())
    if hashes['input'] != digest(source) or hashes['output'] != digest(output):
        raise ValueError('Actor stream provenance mismatch')
    return queries, hashes


def compare(queries, expected, source, failure_path=None):
    """Consume one expected/native frame at a time, retaining no episode arrays."""
    protocol = Protocol(queries)
    verified = 0
    with Path(source).open('rb') as stdin, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen([str(ROOT / 'native/target/release/dynamics-probe')],
                                   stdin=stdin, stdout=subprocess.PIPE, stderr=stderr, text=True)
        timeout = threading.Timer(1800, process.kill)
        timeout.daemon = True
        timeout.start()
        try:
            for row in expected:
                protocol.accept(row)
                line = process.stdout.readline()
                if not line:
                    raise ValueError('Native actor stream ended early')
                actual = json.loads(line)
                if actual != row:
                    if failure_path is not None:
                        Path(failure_path).write_text(encode(dict(expected=row, actual=actual)))
                    difference = first_difference(row, actual)
                    compact = lambda v: dict(type=type(v).__name__, length=len(v)) if isinstance(v, (dict, list)) else v
                    return dict(match=False, verified_records=verified, cases_expected=len(queries),
                                first_failures=[dict(case=row['case'], action=row.get('action'),
                                    difference=[difference[0], compact(difference[1]), compact(difference[2])])])
                verified += 1
            summary = protocol.finish()
            if process.stdout.readline():
                raise ValueError('Native actor stream contains extra records')
            if process.wait() != 0:
                stderr.seek(0)
                raise RuntimeError('Native actor process failed: ' + stderr.read(4096).decode(errors='replace'))
            return dict(match=True, verified_records=verified, **summary, first_failures=[])
        finally:
            timeout.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def verify(directory, queries):
    directory = Path(directory)
    with (directory / 'unity.jsonl').open() as stream:
        return compare(queries, records(stream), directory / 'input.jsonl', directory / 'first-difference.json')


def promote(directory, fixture, queries, selected, hashes, scope):
    selected = sorted(set(selected))
    mapping = {original: i for i, original in enumerate(selected)}
    header = dict(schema=SCHEMA, queries=[queries[i] for i in selected], source_cases=selected,
                  provenance=hashes, scope=scope)
    with Path(fixture).open('wb') as raw, (lzma.LZMAFile(raw, mode='w', preset=3) if Path(fixture).suffix == '.xz' else gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0)) as zipped, io.TextIOWrapper(zipped) as output:
        output.write(encode(header))
        with (Path(directory) / 'unity.jsonl').open() as source:
            for row in records(source):
                if row['case'] in mapping:
                    row['case'] = mapping[row['case']]
                    output.write(encode(row))


def golden(fixture):
    opener = lzma.open if Path(fixture).suffix == '.xz' else gzip.open
    with opener(fixture, 'rt') as stream:
        header = json.loads(next(stream))
        queries = header['queries']
        if header['schema'] != SCHEMA or len(queries) != len(set(header['source_cases'])) or not all(q.get('stream_frames') is True for q in queries):
            raise ValueError('Invalid actor stream fixture')
        with tempfile.TemporaryDirectory(prefix='bopl-actor-stream-') as temporary:
            source = Path(temporary) / 'input.jsonl'
            with source.open('w') as output:
                for query in queries:
                    output.write(encode(query))
            return compare(queries, records(stream), source)
