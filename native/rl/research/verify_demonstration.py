"""Compare tick-for-tick C replay reconstruction with ordinary game playback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from replay_data import ROOT, ReplayWorld, decode
sys.path.insert(0,str(ROOT/'native/tools'))
from reference import read_trace, first_difference, resolve_game, player_sources, provenance

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controlled',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--compare-only',action='store_true')
    parser.add_argument('--game',type=Path,help='Unity player; defaults to BOPL_RL_GAME or the IL2CPP build')
    args=parser.parse_args();out=args.output.resolve();meta,inputs=decode(args.replay)
    player = resolve_game(args.game)
    if args.controlled:
        sources = dict(replay_sha256=args.replay, **player_sources(player))
        before = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
        with ReplayWorld(meta,out/'controlled.log',game=player) as world:
            for t in range(meta['ticks']):
                if world.ended: break
                world.step(list(zip(inputs[t]['angle'],inputs[t]['buttons'])))
        after = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
        if before != after: raise ValueError('Player or replay changed during controlled capture')
        trace = Path(os.environ['BOPL_TRACE'])
        record = dict(before, trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), player=str(player))
        trace.with_suffix('.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        return
    controlled,ordinary=out/'controlled.jsonl.gz',out/'ordinary.jsonl.gz'
    if not args.compare_only:
        out.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,BOPL_TRACE=str(controlled),BOPL_TRACE_TICKS=str(meta['ticks']),BOPL_TRACE_BATCH='16')
        subprocess.run([sys.executable,__file__,str(args.replay.resolve()),'--output',str(out),'--controlled','--game',str(player)],env=env,check=True,timeout=600)
        subprocess.run([sys.executable,str(ROOT/'native/tools/reference.py'),'capture',str(args.replay.resolve()),str(ordinary),
                        '--ticks',str(meta['ticks']),'--batch','16','--timeout','600','--game',str(player)],check=True)
    controlled_identity, ordinary_identity = provenance(controlled), provenance(ordinary)
    for key in ('replay_sha256','player_binary_sha256','unity_player_sha256','game_assembly_sha256','il2cpp_metadata_sha256'):
        if controlled_identity.get(key) != ordinary_identity.get(key):
            raise ValueError(f'Controlled and ordinary capture provenance differs: {key}')
    if controlled_identity['replay_sha256'] != meta['sha256']:
        raise ValueError('Captures do not belong to the requested replay')
    ignored=frozenset(json.loads((ROOT/'native/data/presentation-fields.json').read_text())['fields'])
    left,right=read_trace(controlled),read_trace(ordinary)
    a,b=next(left),next(right)
    if a['setup']!=b['setup']: raise AssertionError('Replay setup differs')
    count=0
    while True:
        a,b=next(left,None),next(right,None)
        if a and b and a['kind']==b['kind']=='end':
            a.pop('reason',None);b.pop('reason',None)
        difference=first_difference(a,b,ignored_fields=ignored)
        if difference: raise AssertionError((count,difference))
        if a is None: break
        if a['kind']=='state':count+=1
    if count!=meta['ticks']: raise AssertionError('Incomplete capture')
    sources=dict(replay=args.replay,controlled=controlled,ordinary=ordinary)
    report=dict(match=True,matched_ticks=count,game_runtime={k:v for k,v in controlled_identity.items() if k not in ('trace_sha256','replay_sha256','player')},projection='gameplay-with-audited-exclusions',excluded_fields=sorted(ignored),
                **{name+'_sha256':hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()})
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
