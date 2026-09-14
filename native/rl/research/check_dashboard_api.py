"""Exercise dashboard HTTP cache invalidation and the ladder API against a live local server."""
import json, subprocess, sys, tempfile, time, urllib.request, urllib.error, socket
from pathlib import Path
serve = Path(__file__).with_name('serve.py').resolve()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    run = root / 'test'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(dict(schema=1, algorithm='recurrent PPO')))
    (run / 'status.json').write_text('{"status":"complete"}')
    file = run / 'metrics.jsonl'
    file.write_text('{"step":1,"map_matches":{"0":3}}\n')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([sys.executable, str(serve), '--root', str(root),
                               '--host', '127.0.0.1', '--port', str(port)],
                              stdout=subprocess.DEVNULL)
    try:
        def open_url(path, etag=None):
            headers = {}
            if etag:
                headers['If-None-Match'] = etag
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        f'http://127.0.0.1:{port}{path}', headers=headers), timeout=30) as response:
                    body = response.read()
                    return response.status, response.headers.get('ETag'), body
            except urllib.error.HTTPError as error:
                return error.code, None, None

        def request(path='/api/run/test', etag=None):
            status, etag_out, body = open_url(path, etag)
            data = json.loads(body) if body else None
            return status, etag_out, data

        for _ in range(200):
            try:
                status, etag, data = request()
                break
            except urllib.error.URLError:
                time.sleep(.05)
        else:
            raise SystemExit('server did not start')
        assert status == 200 and data['unrated'] is True
        assert 'metrics' not in data and data['evaluations'] == []
        assert request(etag=etag)[0] == 304
        metrics_status, metrics_etag, metrics = request('/api/run/test/metrics')
        assert metrics_status == 200 and metrics['metrics_total'] == 1
        assert request('/api/run/test/metrics', metrics_etag)[0] == 304
        with file.open('a') as out:
            out.write('{"step":2,"map_matches":{"0":4}}\n')
        status, new_metrics_etag, metrics = request('/api/run/test/metrics', metrics_etag)
        assert status == 200 and new_metrics_etag != metrics_etag
        assert metrics['metrics_total'] == 2 and metrics['map_matches']['0'] == 4
        assert request('/api/run/test', etag)[0] == 304
        (run / 'status.json').write_text('{"status":"interrupted"}')
        status, etag, data = request(etag=etag)
        assert status == 200 and data['status']['status'] == 'interrupted'
        evaldir = run / 'evaluations' / '000000002'
        evaldir.mkdir(parents=True)
        (evaldir / 'result.json').write_text('{"step":2,"games":[],"groups":[]}')
        status, etag, data = request(etag=etag)
        assert status == 200 and len(data['evaluations']) == 1
        status, _, report = request('/api/run/test/evaluation/000000002')
        assert status == 200 and report['directory'] == '000000002' and report['games'] == []
        file.write_text('{"step":9}\n')
        status, _, metrics = request('/api/run/test/metrics')
        assert status == 200 and metrics['metrics_total'] == 1 and metrics['metrics'][0]['step'] == 9
        status, _, data = request()
        assert status == 200 and 'metrics' not in data

        status, _, body = open_url('/api/index')
        index = json.loads(body)
        assert status == 200 and 'runs' in index and 'families' in index
        assert any(row['name'] == 'test' and row['unrated'] for row in index['runs'])

        status, _, body = open_url('/api/ladder')
        ladder = json.loads(body)
        assert status == 200 and 'rows' in ladder and 'members' in ladder
        if ladder.get('primary') or ladder.get('pools'):
            assert any(m.get('name') == 'random' and m.get('rating') == 0 for m in ladder['members']) or any(
                row.get('name') == 'random' and row.get('rating') == 0 for row in ladder['rows'])
        if ladder.get('human'):
            assert 'interval' in ladder['human'] and ladder['human']['games'] > 0

        status, _, body = open_url('/')
        html = body.decode()
        assert status == 200 and 'id="runRows"' in html and 'id="ladderRows"' in html and 'id="eloChart"' in html
        status, _, body = open_url('/ladder')
        assert status == 200 and 'id="ladderRows"' in body.decode()
        status, _, body = open_url('/run/test')
        assert status == 200 and 'id="runElo"' in body.decode()
        status, _, _ = open_url('/api/sweeps')
        assert status == 200

        evaldir.mkdir(parents=True, exist_ok=True)
        visual = evaldir / 'bot-0.json'
        replay = evaldir / 'bot-0.rep'
        mind = evaldir / 'bot-0.mind.json.gz'
        visual.write_text(json.dumps({
            'schema': 1, 'step': 2, 'task': {'name': 'bot-0'},
            'result': {'win': True}, 'tick_rate': 60,
            'frames': [
                {'t': 0, 'flags': 4, 'players': [{'id': 1, 'flags': 1, 'kills': 0, 'x': -1.23456, 'y': 2.0, 'scale': 1.0}],
                 'entities': [], 'telemetry': [], 'slimes': [], 'actions': []},
                {'t': 4, 'flags': 4, 'players': [{'id': 1, 'flags': 1, 'kills': 0, 'x': 3.14159, 'y': 2.0, 'scale': 1.0}],
                 'entities': [], 'telemetry': [], 'slimes': [], 'actions': []},
            ],
        }))
        replay.write_bytes(b'BOPLREP2')
        with __import__('gzip').open(mind, 'wt') as handle:
            handle.write('{"schema":1}')
        visual_path = '/runs/test/evaluations/000000002/bot-0.json'
        status, etag, body = open_url(visual_path)
        assert status == 200 and etag
        lines = body.decode().strip().split('\n')
        assert len(lines) == 3
        meta, frame0, frame1 = map(json.loads, lines)
        assert 'frames' not in meta and meta['tick_rate'] == 60
        assert frame0['players'][0]['x'] == -1.235
        req = urllib.request.Request(f'http://127.0.0.1:{port}{visual_path}',
                                     headers={'Accept-Encoding': 'gzip'})
        with urllib.request.urlopen(req, timeout=10) as response:
            gz_body = response.read()
            assert response.headers.get('Content-Encoding') == 'gzip'
            assert response.headers.get('Content-Type') == 'application/x-ndjson'
            assert len(gz_body) < visual.stat().st_size
        assert open_url(visual_path, etag)[0] == 304
        visual.write_text(visual.read_text().replace('2.0', '9.5'))
        status, new_etag, body = open_url(visual_path)
        assert status == 200 and new_etag != etag
        assert json.loads(body.decode().strip().split('\n')[1])['players'][0]['y'] == 9.5
        status, _, body = open_url('/runs/test/evaluations/000000002/bot-0.rep')
        assert status == 200 and body == b'BOPLREP2'
        status, _, body = open_url('/runs/test/evaluations/000000002/bot-0.mind.json.gz')
        if body[:2] == b'\x1f\x8b':
            body = __import__('gzip').decompress(body)
        assert status == 200 and json.loads(body)['schema'] == 1
        assert (evaldir / 'bot-0.json.dash.gz').is_file()
        print('PASS: cache invalidation, index/ladder payload, human/pool rows, HTML routes, visual ndjson')
    finally:
        server.terminate()
        server.wait(timeout=5)
