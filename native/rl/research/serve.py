"""Local research dashboard with on-demand saved replay playback."""
import argparse
from collections import OrderedDict
import glob
import hashlib
import threading
import gzip
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
import mimetypes
import re
import os
from pathlib import Path
import tempfile
from urllib.parse import unquote,urlparse,parse_qs
import run_insights
from metrics_store import store
import study_data
from replay_library import ReplayLibrary

HERE=Path(__file__).resolve().parent
def read_json(path,default=None):
    return study_data.read(path,default)


def compact_report(path):
    stat=path.stat()
    # Reports are immutable after publication. Cache their compact view only.
    value=_compact_report(path,stat.st_ino,stat.st_mtime_ns,stat.st_size)
    return dict(value,games=[dict(game) for game in value['games']])


@lru_cache(maxsize=256)
def _compact_report(path,inode,modified,size):
    report=json.loads(path.read_text())
    if not report: return None
    # The full provenance and telemetry remain available in result.json.
    value={key:report[key] for key in ('schema','step','suite','groups','frozen_opponents','flags','flag_metrics') if key in report}
    value['games']=[{key:game[key] for key in ('name','opponent','map','seed','side','save','win','loss','timeout','kills','deaths','ticks','replay','visual','mind') if key in game} for game in report.get('games',[])]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE.parents[1]/'artifacts/rl')
    parser.add_argument('--host',default='127.0.0.1'); parser.add_argument('--port',type=int,default=8876)
    parser.add_argument('--study-root',action='append',default=[],metavar='NAME=PATH',help='Read a live study from an isolated workspace')
    parser.add_argument('--run-root',action='append',default=[],metavar='NAME=PATH',help='Read a live training run directory from an isolated workspace')
    parser.add_argument('--run-root-glob',action='append',default=[],metavar='NAME=GLOB',help='Register every run directory matching GLOB as NAME-<dirname> (e.g. a sweep tree), re-evaluated per request')
    parser.add_argument('--ladder',action='append',default=[],metavar='NAME=PATH',help='Directory holding ladder.json (elo_ladder/reference_pool schema) and optionally rescore.json (sweep/rescore.py) for the Ladder view')
    args=parser.parse_args(); root=args.root.resolve()
    library=ReplayLibrary(root)
    extra={};extra_runs={};run_globs={};ladders={}
    for attribute,target in (('study_root',extra),('run_root',extra_runs),('run_root_glob',run_globs),('ladder',ladders)):
        for value in getattr(args,attribute):
            name,separator,path=value.partition('=')
            if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+',name):parser.error(f'{attribute} requires NAME=PATH')
            target[name]=path if attribute=='run_root_glob' else Path(path).resolve()
    def named_runs():
        """Explicit --run-root entries plus live matches of every --run-root-glob."""
        found=dict(extra_runs)
        for prefix,pattern in run_globs.items():
            for match in sorted(glob.glob(pattern)):
                directory=Path(match)
                if directory.is_dir() and re.fullmatch(r'[A-Za-z0-9_.-]+',directory.name):
                    found.setdefault(f'{prefix}-{directory.name}',directory.resolve())
        return found
    def runs():
        candidates=[];named=named_runs()
        for file in [*root.glob('*/config.json'),*(path/'config.json' for path in named.values())]:
            try: config=read_json(file,{})
            except (OSError,ValueError): continue
            if config and config.get('schema') in (1,2) and config.get('algorithm')=='recurrent PPO':
                name=next((n for n,p in named.items() if p==file.parent),file.parent.name)
                candidates.append(dict(name=name,created=file.stat().st_mtime,
                                       workspace=str(file.parent) if name in named else None,
                                       status=read_json(file.parent/'status.json',{})))
        return sorted(candidates,key=lambda r:r['created'],reverse=True)
    def run_directory(name):
        named=named_runs()
        if name in named: return named[name]
        if re.fullmatch(r'[A-Za-z0-9_.-]+',name) and (root/name/'config.json').is_file(): return root/name
        return None
    def step_order(path):
        # Evaluation/checkpoint directories are named by step, zero-padded to 9 digits;
        # runs past 1e9 game steps break lexical order, so sort numerically.
        match=re.search(r'(\d+)',path.parent.name if path.name=='result.json' else path.stem)
        return (int(match.group(1)) if match else -1, str(path))
    def latest_report(directory, pattern):
        reports=sorted(directory.glob(pattern), key=step_order)
        return compact_report(reports[-1]) if reports else None
    def run_payload(directory,name):
        status=dict(read_json(directory/'status.json',{'status':'starting'}))
        if status.get('status')=='running':
            try: os.kill(status['pid'],0)
            except (ProcessLookupError,KeyError): status['status']='interrupted'
        try: summary=store.read(directory/'metrics.jsonl')
        except ValueError: raise ValueError('Invalid metrics record')
        metrics=summary['rows']
        evaluations=[]
        for file in sorted(directory.glob('evaluations/*/result.json'), key=step_order):
            report=compact_report(file); report['directory']=file.parent.name
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
        return dict(name=name,config=read_json(directory/'config.json'),status=status,
                              metrics=metrics,metrics_total=summary['count'],map_matches=summary['map_matches'],
                              evaluations=evaluations,video=read_json(directory/'video-status.json',{}),
                              insights=insights,ladder=read_json(directory/'ladder/ladder.json'),
                              final_evaluation=latest_report(directory,'final-evaluations/*/result.json'),
                              validation_evaluation=latest_report(directory,'validation/evaluations/*/result.json'))

    def stamp(path):
        try: value=path.stat()
        except FileNotFoundError: return None
        return value.st_ino,value.st_mtime_ns,value.st_size

    responses=OrderedDict()
    response_lock=threading.Lock()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*args):
            if len(args)<2 or str(args[1]) not in ('200','206','304'): super().log_message(fmt,*args)
        def json(self,data):
            payload=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            if len(payload)>16384 and 'gzip' in self.headers.get('Accept-Encoding',''):
                payload=gzip.compress(payload,1)
                self.send_header('Content-Encoding','gzip')
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            if self.command!='HEAD':
                try: self.wfile.write(payload)
                except (BrokenPipeError,ConnectionResetError): pass
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
            if path=='/api/ladders':
                return self.json({name:dict(ladder=read_json(directory/'ladder.json'),rescore=read_json(directory/'rescore.json'))
                                  for name,directory in ladders.items() if (directory/'ladder.json').is_file()})
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
                                    for p in sorted(directory.glob('evaluations/*/result.json'), key=step_order) if (r:=read_json(p))]
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
                try:
                    with response_lock:
                        dependencies=[directory/file for file in ('config.json','status.json','metrics.jsonl','video-status.json','ladder/ladder.json')]
                        for pattern in ('evaluations/*/result.json','final-evaluations/*/result.json','validation/evaluations/*/result.json'):
                            for result in sorted(directory.glob(pattern)):
                                dependencies.extend((result,result.parent))
                        status=read_json(directory/'status.json',{})
                        alive=True
                        if status.get('status')=='running':
                            try: os.kill(status['pid'],0)
                            except (ProcessLookupError,KeyError): alive=False
                        signature=(alive,tuple((str(p),stamp(p)) for p in dependencies))
                        cached=responses.pop(name,None)
                        if cached is None or cached[0]!=signature:
                            payload=json.dumps(run_payload(directory,name),allow_nan=False,separators=(',',':')).encode()
                            cached=(signature,'"'+hashlib.sha256(payload).hexdigest()+'"',payload,gzip.compress(payload,1))
                        else: run_insights.collect(directory)
                        responses[name]=cached
                        while len(responses)>4: responses.popitem(last=False)
                except ValueError: return self.send_error(500,'Invalid metrics record')
                _,etag,payload,compressed=cached
                if self.headers.get('If-None-Match')==etag:
                    self.send_response(304); self.send_header('ETag',etag); self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Cache-Control','no-cache')
                self.send_header('ETag',etag)
                self.send_header('Vary','Accept-Encoding')
                if 'gzip' in self.headers.get('Accept-Encoding',''):
                    payload=compressed; self.send_header('Content-Encoding','gzip')
                self.send_header('Content-Length',str(len(payload))); self.end_headers()
                if self.command!='HEAD':
                    try: self.wfile.write(payload)
                    except (BrokenPipeError,ConnectionResetError): pass
                return
            if path.startswith('/studies/'):
                parts=path[len('/studies/'):].split('/')
                if not parts or parts[0] not in studies:return self.send_error(404)
                study_root=studies[parts[0]].resolve();file=(study_root/'/'.join(parts[1:])).resolve()
                if not file.is_relative_to(study_root) or file.suffix not in ('.json','.jsonl','.rep','.mp4','.bin','.pt','.jpg','.png','.svg','.pdf','.md','.gz'):
                    return self.send_error(403)
            elif path.startswith('/runs/'):
                parts=path[len('/runs/'):].split('/')
                directory=run_directory(parts[0]) if parts else None
                if directory is None: return self.send_error(404)
                file=(directory/'/'.join(parts[1:])).resolve()
                # .bin serves native checkpoints; .pt stays readable for historical runs.
                if not file.is_relative_to(directory) or file.suffix not in ('.json','.jsonl','.rep','.mp4','.bin','.pt','.jpg','.gz'):
                    return self.send_error(403)
            elif path in ('/','/index.html','/app.js','/style.css','/study','/study.html','/study.js','/study.css','/ladder','/ladder.html','/ladder.js','/arena.js','/matches.js'):
                query=parse_qs(urlparse(self.path).query)
                # A live training run is the front page; the study catalog is the home only when nothing is running.
                live=any((r.get('status') or {}).get('status')=='running' for r in runs())
                study_home=path=='/' and studies and not live and not any(k in query for k in ('run','series','history'))
                file=HERE/'web'/('study.html' if study_home or path=='/study' else 'ladder.html' if path=='/ladder' else 'index.html' if path=='/' else path[1:])
            else: return self.send_error(404)
            if not file.is_file(): return self.send_error(404)
            size=file.stat().st_size; start=0; end=size-1
            requested=self.headers.get('Range')
            # Replay JSON grows with episode length (a full 7200-tick self-play match is ~12 MB);
            # compress whole-file JSON responses for clients that accept it, as the API does.
            if not requested and file.suffix in ('.json','.jsonl') and size>16384 and 'gzip' in self.headers.get('Accept-Encoding',''):
                compressed=tempfile.TemporaryFile()
                with file.open('rb') as source, gzip.GzipFile(fileobj=compressed,mode='wb',compresslevel=1) as target:
                    import shutil
                    shutil.copyfileobj(source,target,65536)
                compressed_size=compressed.tell()
                compressed.seek(0)
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Encoding','gzip')
                self.send_header('Content-Length',str(compressed_size))
                self.end_headers()
                if self.command!='HEAD':
                    try: shutil.copyfileobj(compressed,self.wfile,65536)
                    except (BrokenPipeError,ConnectionResetError): pass
                compressed.close()
                return
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
