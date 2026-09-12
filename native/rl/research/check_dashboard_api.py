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
                        f'http://127.0.0.1:{port}{path}', headers=headers), timeout=10) as response:
                    body = response.read()
                    return response.status, response.headers.get('ETag'), body
            except urllib.error.HTTPError as error:
                return error.code, None, None

        def request(etag=None):
            status, etag_out, body = open_url('/api/run/test', etag)
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
        assert status == 200 and data['metrics_total'] == 1
        assert data['unrated'] is True
        assert request(etag)[0] == 304
        with file.open('a') as out:
            out.write('{"step":2,"map_matches":{"0":4}}\n')
        status, new_etag, data = request(etag)
        assert status == 200 and new_etag != etag and data['metrics_total'] == 2 and data['map_matches']['0'] == 4
        assert request(new_etag)[0] == 304
        (run / 'status.json').write_text('{"status":"interrupted"}')
        status, etag, data = request(new_etag)
        assert status == 200 and data['status']['status'] == 'interrupted'
        evaldir = run / 'evaluations' / '000000002'
        evaldir.mkdir(parents=True)
        (evaldir / 'result.json').write_text('{"step":2,"games":[],"groups":[]}')
        status, etag, data = request(etag)
        assert status == 200 and len(data['evaluations']) == 1
        file.write_text('{"step":9}\n')
        status, etag, data = request(etag)
        assert status == 200 and data['metrics_total'] == 1 and data['metrics'][0]['step'] == 9

        status, _, body = open_url('/api/ladder')
        ladder = json.loads(body)
        assert status == 200 and 'rows' in ladder and 'members' in ladder
        assert any(row['name'] == 'test' and row['unrated'] for row in ladder['rows'])
        if ladder.get('pool'):
            assert any(m['name'] == 'random' and m['rating'] == 0 for m in ladder['members'])
        if ladder.get('human'):
            assert 'interval' in ladder['human'] and ladder['human']['games'] > 0

        status, _, body = open_url('/')
        html = body.decode()
        assert status == 200 and 'id="ladderRows"' in html and 'id="eloChart"' in html
        status, _, body = open_url('/run/test')
        assert status == 200 and 'id="runElo"' in body.decode()
        status, _, _ = open_url('/api/sweeps')
        assert status == 200
        print('PASS: cache invalidation, ladder payload, human/pool rows, HTML routes')
    finally:
        server.terminate()
        server.wait(timeout=5)
