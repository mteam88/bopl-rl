"""Read-only local dashboard for research runs, metrics, and replays."""
import argparse
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import re
import os
from pathlib import Path
from urllib.parse import unquote,urlparse

HERE=Path(__file__).resolve().parent

def read_json(path,default=None):
    try: return json.loads(path.read_text())
    except FileNotFoundError: return default

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE.parents[1]/'artifacts/rl')
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8876)
    args=parser.parse_args(); root=args.root.resolve()
    def runs():
        candidates=[]
        for file in root.glob('*/config.json'):
            try: config=read_json(file,{})
            except (OSError,ValueError): continue
            if config.get('schema')==1 and config.get('algorithm')=='recurrent PPO':
                candidates.append(dict(name=file.parent.name,created=file.stat().st_mtime,
                                       status=read_json(file.parent/'status.json',{})))
        return sorted(candidates,key=lambda r:r['created'],reverse=True)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*args):
            if len(args)<2 or str(args[1]) not in ('200','206','304'): super().log_message(fmt,*args)
        def json(self,data):
            payload=json.dumps(data,allow_nan=False).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            if self.command!='HEAD': self.wfile.write(payload)
        def do_GET(self):
            path=unquote(urlparse(self.path).path)
            if path=='/api/runs': return self.json(runs())
            if path.startswith('/api/run/'):
                name=path[len('/api/run/'):]
                if name not in {r['name'] for r in runs()}: return self.send_error(404)
                directory=root/name; status=read_json(directory/'status.json',{'status':'starting'})
                if status.get('status')=='running':
                    try: os.kill(status['pid'],0)
                    except (ProcessLookupError,KeyError): status['status']='interrupted'
                metrics=[]
                file=directory/'metrics.jsonl'
                if file.exists():
                    lines=file.read_text().splitlines()
                    for i,line in enumerate(lines):
                        try: metrics.append(json.loads(line))
                        except ValueError:
                            if i!=len(lines)-1: return self.send_error(500,'Invalid metrics record')
                evaluations=[]
                for file in sorted(directory.glob('evaluations/*/result.json')):
                    report=read_json(file); report['directory']=file.parent.name
                    for game in report['games']:
                        video=file.parent/(game['name']+'.mp4')
                        if video.exists(): game['video']=video.name
                        poster=video.with_suffix('.jpg')
                        if poster.exists(): game['poster']=poster.name
                    evaluations.append(report)
                return self.json(dict(name=name,config=read_json(directory/'config.json'),status=status,
                                      metrics=metrics,evaluations=evaluations,video=read_json(directory/'video-status.json',{})))
            if path.startswith('/runs/'):
                parts=path[len('/runs/'):].split('/')
                if not parts or parts[0] not in {r['name'] for r in runs()}: return self.send_error(404)
                file=(root/'/'.join(parts)).resolve()
                if not file.is_relative_to(root/parts[0]) or file.suffix not in ('.json','.jsonl','.rep','.mp4','.pt','.jpg'):
                    return self.send_error(403)
            elif path in ('/','/index.html','/app.js','/style.css'):
                file=HERE/'web'/('index.html' if path=='/' else path[1:])
            else: return self.send_error(404)
            if not file.is_file(): return self.send_error(404)
            size=file.stat().st_size; start=0; end=size-1
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
                if not match or not any(match.groups()): return self.send_error(416)
                a,b=match.groups()
                if a:
                    start=int(a); end=min(end,int(b)) if b else end
                else: start=max(0,size-int(b))
                if start>end or start>=size: return self.send_error(416)
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type',mimetypes.guess_type(file)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(max(0,end-start+1)))
            if requested: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD': return
            with file.open('rb') as stream:
                stream.seek(start); remaining=end-start+1
                while remaining>0:
                    chunk=stream.read(min(65536,remaining))
                    if not chunk: break
                    try: self.wfile.write(chunk)
                    except (BrokenPipeError,ConnectionResetError): break
                    remaining-=len(chunk)
        def do_HEAD(self): return self.do_GET()
        def end_headers(self):
            self.send_header('X-Content-Type-Options','nosniff')
            super().end_headers()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Bopl research dashboard: http://{args.host}:{args.port}',flush=True)
    server.serve_forever()

if __name__=='__main__': main()
