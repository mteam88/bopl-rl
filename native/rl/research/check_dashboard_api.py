"""Exercise dashboard HTTP cache invalidation against a live local server."""
import json,subprocess,sys,tempfile,time,urllib.request,urllib.error,socket
from pathlib import Path
serve=Path(__file__).with_name('serve.py').resolve()
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);run=root/'test';run.mkdir()
    (run/'config.json').write_text(json.dumps(dict(schema=1,algorithm='recurrent PPO')))
    (run/'status.json').write_text('{"status":"complete"}')
    file=run/'metrics.jsonl';file.write_text('{"step":1,"map_matches":{"0":3}}\n')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    server=subprocess.Popen([sys.executable,str(serve),'--root',str(root),'--port',str(port)],stdout=subprocess.DEVNULL)
    try:
        url=f'http://127.0.0.1:{port}/api/run/test'
        def request(etag=None):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'If-None-Match':etag} if etag else {}),timeout=5) as response:
                    return response.status,response.headers['ETag'],json.load(response)
            except urllib.error.HTTPError as error:return error.code,None,None
        for _ in range(50):
            try:status,etag,data=request();break
            except urllib.error.URLError:time.sleep(.05)
        assert status==200 and data['metrics_total']==1
        assert request(etag)[0]==304
        with file.open('a') as out:out.write('{"step":2,"map_matches":{"0":4}}\n')
        status,new_etag,data=request(etag)
        assert status==200 and new_etag!=etag and data['metrics_total']==2 and data['map_matches']['0']==4
        assert request(new_etag)[0]==304
        (run/'status.json').write_text('{"status":"interrupted"}')
        status,etag,data=request(new_etag);assert status==200 and data['status']['status']=='interrupted'
        evaldir=run/'evaluations'/'000000002';evaldir.mkdir(parents=True)
        (evaldir/'result.json').write_text('{"step":2,"games":[],"groups":[]}')
        status,etag,data=request(etag);assert status==200 and len(data['evaluations'])==1
        file.write_text('{"step":9}\n')
        status,etag,data=request(etag);assert status==200 and data['metrics_total']==1 and data['metrics'][0]['step']==9
        print('PASS: HTTP conditional requests invalidate on append, status, evaluation publication and truncation')
    finally:server.terminate();server.wait(timeout=5)
