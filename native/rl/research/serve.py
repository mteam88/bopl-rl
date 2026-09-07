"""Local research dashboard with on-demand saved replay playback."""
import argparse
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import re
import os
from pathlib import Path
from urllib.parse import unquote,urlparse,parse_qs
import run_insights
import study_data
from replay_library import ReplayLibrary

HERE=Path(__file__).resolve().parent

def read_json(path,default=None):
    try: return json.loads(path.read_text())
    except FileNotFoundError: return default

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE.parents[1]/'artifacts/rl')
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8876)
    parser.add_argument('--study-root',action='append',default=[],metavar='NAME=PATH',help='Read a live study from an isolated workspace')
    parser.add_argument('--run-root',action='append',default=[],metavar='NAME=PATH',help='Read a live training run directory from an isolated workspace')
    args=parser.parse_args(); root=args.root.resolve()
    library=ReplayLibrary(root)
    extra={};extra_runs={}
    for attribute,target in (('study_root',extra),('run_root',extra_runs)):
        for value in getattr(args,attribute):
            name,separator,path=value.partition('=')
            if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+',name):parser.error(f'{attribute} requires NAME=PATH')
            target[name]=Path(path).resolve()
    def runs():
        candidates=[]
        for file in [*root.glob('*/config.json'),*(path/'config.json' for path in extra_runs.values())]:
            try: config=read_json(file,{})
            except (OSError,ValueError): continue
            if config and config.get('schema')==1 and config.get('algorithm')=='recurrent PPO':
                name=next((n for n,p in extra_runs.items() if p==file.parent),file.parent.name)
                candidates.append(dict(name=name,created=file.stat().st_mtime,
                                       workspace=str(file.parent) if name in extra_runs else None,
                                       status=read_json(file.parent/'status.json',{})))
        return sorted(candidates,key=lambda r:r['created'],reverse=True)
    def run_directory(name):
        if name in extra_runs: return extra_runs[name]
        if re.fullmatch(r'[A-Za-z0-9_.-]+',name) and (root/name/'config.json').is_file(): return root/name
        return None
    def latest_report(directory, pattern):
        reports=sorted(directory.glob(pattern))
        return read_json(reports[-1]) if reports else None
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*args):
            if len(args)<2 or str(args[1]) not in ('200','206','304'): super().log_message(fmt,*args)
        def json(self,data):
            payload=json.dumps(data,allow_nan=False).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            if self.command!='HEAD': self.wfile.write(payload)
        def replay_request(self, path, prepare=False):
            parts=path.split('/')
            studies=study_data.catalog(root,extra)
            if len(parts)!=5 or parts[3] not in studies:return self.send_error(404)
            try:return self.json(library.replay(parts[3],studies[parts[3]],parts[4],prepare))
            except KeyError:return self.send_error(404)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send_error(403)
            path=unquote(urlparse(self.path).path)
            if path.startswith('/api/replay/'):return self.replay_request(path,True)
            return self.send_error(404)
        def do_GET(self):
            path=unquote(urlparse(self.path).path)
            studies=study_data.catalog(root,extra)
            if path.startswith('/api/replay/'):return self.replay_request(path)
            if path.startswith('/api/matches/'):
                name=path[len('/api/matches/'):]
                if name not in studies:return self.send_error(404)
                return self.json(library.catalog(name,studies[name]))
            if path=='/api/studies':
                return self.json([dict(name=name,title='A/B continuation' if 'total_learner_target' in study_data.read(directory/'protocol.json') else 'Scripted opponents + Bow initialization') for name,directory in reversed(list(studies.items()))])
            if path.startswith('/api/study/'):
                name=path[len('/api/study/'):]
                if name not in studies:return self.send_error(404)
                return self.json(study_data.payload(name,studies[name],root))
            if path=='/api/runs': return self.json(runs())
            if path=='/api/series':
                series=[]
                for file in sorted(root.glob('*/plan.json')):
                    plan=read_json(file,{})
                    if plan.get('kind')!='bopl-training-series': continue
                    members=[]
                    for member in plan['runs']:
                        name=member['name']
                        if not re.fullmatch(r'[A-Za-z0-9_-]+',name): continue
                        directory=root/name
                        diagnostic=[dict(step=r['step'],groups=r['groups'])
                                    for p in sorted(directory.glob('evaluations/*/result.json')) if (r:=read_json(p))]
                        final=latest_report(directory,'final-evaluations/*/result.json')
                        members.append(dict(**member,status=read_json(directory/'status.json',{'status':'queued'}),
                                            evaluations=diagnostic,final_groups=final['groups'] if final else None))
                    series.append(dict(name=plan['name'],steps=plan['steps'],runs=members,
                                       status=read_json(file.parent/'status.json',{})))
                return self.json(series)
            if path.startswith('/api/insights/'):
                parts=path[len('/api/insights/'):].split('/')
                directory=run_directory(parts[0]) if len(parts)==2 else None
                if not directory or not re.fullmatch(r'\d+',parts[1]): return self.send_error(404)
                value=read_json(directory/'evaluations'/parts[1]/'insights.json')
                if not value: return self.send_error(404)
                return self.json(value)
            if path.startswith('/api/run/'):
                name=path[len('/api/run/'):]
                directory=run_directory(name)
                if directory is None: return self.send_error(404)
                status=read_json(directory/'status.json',{'status':'starting'})
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
                        mind=file.parent/(game['name']+'.mind.json.gz')
                        if mind.exists(): game['mind']=mind.name
                    evaluations.append(report)
                insights=[{key:row[key] for key in ('step','directory','games','gauges')}
                          for row in run_insights.collect(directory)]
                return self.json(dict(name=name,config=read_json(directory/'config.json'),status=status,
                                      metrics=metrics,evaluations=evaluations,video=read_json(directory/'video-status.json',{}),
                                      insights=insights,ladder=read_json(directory/'ladder/ladder.json'),
                                      final_evaluation=latest_report(directory,'final-evaluations/*/result.json'),
                                      validation_evaluation=latest_report(directory,'validation/evaluations/*/result.json')))
            if path.startswith('/studies/'):
                parts=path[len('/studies/'):].split('/')
                if not parts or parts[0] not in studies:return self.send_error(404)
                study_root=studies[parts[0]].resolve();file=(study_root/'/'.join(parts[1:])).resolve()
                if not file.is_relative_to(study_root) or file.suffix not in ('.json','.jsonl','.rep','.mp4','.pt','.jpg','.png','.svg','.pdf','.md','.gz'):
                    return self.send_error(403)
            elif path.startswith('/runs/'):
                parts=path[len('/runs/'):].split('/')
                directory=run_directory(parts[0]) if parts else None
                if directory is None: return self.send_error(404)
                file=(directory/'/'.join(parts[1:])).resolve()
                if not file.is_relative_to(directory) or file.suffix not in ('.json','.jsonl','.rep','.mp4','.pt','.jpg','.gz'):
                    return self.send_error(403)
            elif path in ('/','/index.html','/app.js','/style.css','/study','/study.html','/study.js','/study.css','/arena.js','/matches.js'):
                query=parse_qs(urlparse(self.path).query)
                study_home=path=='/' and studies and not any(k in query for k in ('run','series','history'))
                file=HERE/'web'/('study.html' if study_home or path=='/study' else 'index.html' if path=='/' else path[1:])
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
            self.send_header('Content-Type','application/json' if file.name.endswith('.json.gz') else mimetypes.guess_type(file)[0] or 'application/octet-stream')
            if file.name.endswith('.json.gz'):self.send_header('Content-Encoding','gzip')
            if file.suffix in ('.html','.js','.css'):self.send_header('Cache-Control','no-cache')
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
