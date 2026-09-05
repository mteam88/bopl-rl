"""Compare complete SpriteAnimator state, frame selection and reentrant callbacks."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import random
import subprocess
from reference import ROOT,first_difference
from verify_beam import encoded,reference
from verify_player_ground import vec
from verify_dynamics import ONE

FIXTURE=ROOT/'native/fixtures/animator.jsonl.gz'
SCHEMA='bopl-animator-v1'


def queries(count,seed):
    rng=random.Random(seed)
    for case in range(count):
        def frames(length=None):
            return [dict(sprite=rng.choice([None,0,1,2,3,4,5,6,7]),duration=rng.choice([-4,-2,-1,0,1,2,4])) for _ in range(rng.randint(1,5) if length is None else length)]
        def maps():
            return [[{'sprite':(i+j)%8,'from':(10-j*10)*ONE,'to':(20-j*10)*ONE} for j in range(rng.randint(1,4))] for i in range(rng.randint(1,4))]
        def tracks():
            return [dict(frames=frames(rng.randint(1,4) if case%17==0 else 4),duration=rng.choice([-2,-1,0,1,2])) for _ in range(rng.randint(1,4))]
        current=frames();velocity=maps();two_d=tracks()
        if case%37==0:current[0]['duration']=-(1<<31)
        state=dict(render=None if case%31==0 else dict(sprite=rng.randrange(8),flip_x=bool(case%2),flip_y=bool(case%3)),
            current=None if case%41==0 else current,next=frames() if case%3==0 else None,index=0,
            position=(1<<31)-1 if case%43==0 else rng.randrange(6),kind=case%7,swapped=bool(case%2),callback=rng.choice([None,1,2,3,4]),
            body=None if case%23==0 else case%3,velocity_maps=velocity,tracks=two_d,position_y=0,going_right=bool(case%2),target=len(two_d)-1,
            x_speeds=[-30*ONE,0,30*ONE],min_x=-100*ONE,swap_timing=rng.randint(-2,5),callback_last=bool(case%2),
            always_stop=case%5==0,position_2d=(1<<31)-1 if case%47==0 else rng.randrange(8),animation_index=0,child_suffix=None)
        if case%53==0:state['velocity_maps']=None
        if case%59==0:state['velocity_maps']=[None]
        if case%61==0:state['tracks']=None
        callbacks={'1':[dict(op='begin',frames=frames(2),next=None,callback=None,style=0)],'2':[],
            '3':[dict(op='begin',frames=frames(2),next=frames(1),callback=2,style=1)],
            '4':[dict(op='begin',frames=[],next=None,callback=None,style=0)]}
        bodies=[dict(velocity=vec(rng.randint(-60,60),rng.randint(-30,30)),rotation=rng.randint(-720*ONE,720*ONE)) for _ in range(3)]
        players=[dict(id=i+1,protected=(case+i)%3==0) for i in range(case%6+1)]
        holder=None if case%11==0 else 99 if case%13==0 else case%len(players)+1
        actions=[dict(op="speed_up",frames=frames(),factor=4)]
        if state['render'] is None:actions.append(dict(op='begin',frames=frames(),next=frames(),callback=3,style=1))
        def ticks(n):
            for i in range(n):
                actions.append(dict(op='update',scale=[0,42949671,42949672,ONE][i%4],dt=rng.choice([-ONE,0,ONE//60,ONE])))
                if i%3==0:actions.append(dict(op='animate'))
        ticks(5)
        for i in range(4):
            style=(case+i)%4
            actions.append(dict(op='begin',frames=frames(3),next=frames(2),callback=rng.choice([1,2,3]),style=style));ticks(8)
        actions.append(dict(op='y',map=maps()[0],looping=frames(2),anticipation=None if case%2 else 7,body=case%3));ticks(7)
        for immediate in [False,True]:
            actions.append(dict(op='xy',maps=maps(),speeds=[-30*ONE,0,30*ONE],min_x=-100*ONE,timing=rng.randint(-2,5),anticipation=None if case%2 else 5,body=case%3,flip=bool(case%2),immediate=immediate))
            actions.append(dict(op='speed',index=case%3,velocity=vec(rng.randint(-90,90),rng.randint(-30,30)),rotation=rng.randint(-720*ONE,720*ONE)))
            ticks(7);actions.append(dict(op='left'))
        new_tracks=tracks()
        if case%79==0:new_tracks[0]['frames'][0]['duration']=-(1<<31)
        if case%73==0:new_tracks[0]['duration']=-(1<<31)
        actions.append(dict(op='two_d',tracks=new_tracks,start_x=0,start_y=0,right=bool(case%2),target=len(new_tracks)-1,override_sprite=bool(case%3),anticipation=None if case%2 else 3));ticks(7)
        actions.append(dict(op='target',index=0));ticks(7)
        actions.append(dict(op='rotation',frames=[dict(sprite=i,duration=-90 if (case+i)%2 else 90) for i in range(4)],body=case%3));ticks(6)
        actions.append(dict(op='speed_up',frames=[dict(sprite=None,duration=-(1<<31) if case%19==0 else -7)],factor=0 if case%23==0 else -2))
        actions.extend([dict(op='child',suffix='spawn'),dict(op='current_sprite'),dict(op='current_index'),dict(op='passed'),dict(op='current_animation'),
            dict(op='skip',index=(1<<31)-1 if case%67==0 else rng.choice([-1,0,2,4])),dict(op='passed'),dict(op='current_sprite')])
        yield dict(operation='dynamics_animator',state=state,bodies=bodies,players=players,holder=holder,callbacks=callbacks,actions=actions)


def asset_queries():
    """Use exported Unity frames, including spawn, and retain exact sprite identity."""
    path=ROOT/'native/reference-export/assets.json'
    raw=path.read_bytes();assets=json.loads(raw);sprites={};queries_out=[]
    template=next(queries(2,326741));template['holder']=1
    template['players']=[dict(id=1,protected=True)];template['bodies']=[dict(velocity=vec(0,0),rotation=0)]
    template['callbacks']={'2':[]}
    template['state'].update(render=dict(sprite=None,flip_x=False,flip_y=False),current=None,next=None,
        index=0,position=0,kind=0,swapped=False,callback=None,body=0,velocity_maps=None,tracks=None,
        position_y=0,going_right=False,target=0,x_speeds=None,min_x=0,swap_timing=0,callback_last=False,
        always_stop=False,position_2d=0,animation_index=0,child_suffix=None)
    def sprite(value):
        if value is None:return None
        if value not in sprites:sprites[value]=len(sprites)
        return sprites[value]
    counts=Counter()
    for asset in assets.values():
        if asset['data'].get('type')!='AnimationData':continue
        counts['assets']+=1;p=asset['data']['properties']
        for i in range(p['animations.Array.size']):
            prefix=f'animations.Array.data[{i}]';name=p[prefix+'.name'];frames=[]
            for j in range(p[prefix+'.animation.Array.size']):
                f=prefix+f'.animation.Array.data[{j}]'
                frames.append(dict(sprite=sprite(p[f+'.sprite']),duration=p[f+'.frameDuration']))
            counts['animations']+=1;counts['frames']+=len(frames)
            q=deepcopy(template);q['asset']=dict(path=asset['path'],animation=name);q['actions']=[]
            for style in range(4):
                q['actions'].append(dict(op='begin',frames=frames,next=frames,callback=2,style=style))
                q['actions'].extend(dict(op='update',scale=ONE,dt=ONE//60) for _ in range(min(256,2*sum(abs(f['duration']) for f in frames)+3)))
            q['actions'].append(dict(op='speed_up',frames=frames,factor=4))
            fast=[dict(sprite=f['sprite'],duration=max(1,abs(f['duration'])//4)*(-1 if f['duration']<0 else 1)) for f in frames]
            q['actions'].append(dict(op='begin',frames=fast,next=None,callback=2,style=3))
            q['actions'].extend(dict(op='animate') for _ in range(min(256,2*sum(abs(f['duration']) for f in fast)+3)))
            queries_out.append(q)
            # Long holds need boundary snapshots without millions of idle calls.
            for j,f in enumerate(frames):
                q=deepcopy(template);q['asset']=dict(path=asset['path'],animation=name,frame=j)
                q['state'].update(current=frames,index=j,position=max(0,abs(f['duration'])-1),callback=2)
                q['actions']=[dict(op='animate') for _ in range(3)];queries_out.append(q)
        for i in range(p['velocityAnimations.Array.size']):
            prefix=f'velocityAnimations.Array.data[{i}]';frames=[]
            for j in range(p[prefix+'.animation.Array.size']):
                f=prefix+f'.animation.Array.data[{j}]'
                frames.append({'sprite':sprite(p[f+'.sprite']),'from':p[f+'.speedFrom.m_rawValue'],'to':p[f+'.speedTo.m_rawValue']})
            counts['velocity_maps']+=1;q=deepcopy(template)
            q['asset']=dict(path=asset['path'],velocity_animation=p[prefix+'.name'])
            q['state'].update(kind=2,velocity_maps=[frames],callback=2)
            q['actions']=[]
            for value in sorted({f[k]+d for f in frames for k in ['from','to'] for d in [-1,0,1]}):
                q['actions'].extend([dict(op='speed',index=0,velocity={'x':0,'y':value},rotation=0),dict(op='animate')])
            queries_out.append(q)
        if p['animation2Ds.Array.size']:raise ValueError('Add extraction for newly added 2D asset tracks')
    return queries_out,dict(export_sha256=hashlib.sha256(raw).hexdigest(),**counts,sprites=len(sprites))


def check(records):
    p=subprocess.run([str(ROOT/'native/target/release/dynamics-probe')],input=encoded(r['query'] for r in records),text=True,capture_output=True,check=True,timeout=300)
    outputs=[json.loads(line) for line in p.stdout.splitlines()]
    if len(outputs)!=len(records):raise ValueError('Incomplete animator output')
    counts,actions,kinds,errors=(Counter() for _ in range(4));failures=[]
    for i,(r,out) in enumerate(zip(records,outputs)):
        expected=r['expected']
        if isinstance(expected,list):frames=expected;counts['complete_scenarios']+=1
        else:frames=expected['completed'];errors[expected['error']]+=1
        counts['completed_actions']+=len(frames)
        actions.update(a['op'] for a in r['query']['actions'][:len(frames)])
        kinds.update(f['state']['kind'] for f in frames)
        counts['callbacks']+=sum(len(f['events']) for f in frames)
        if out!=expected:
            counts['mismatches']+=1
            if len(failures)<10:
                diff=first_difference(expected,out);compact=lambda v:dict(type=type(v).__name__,length=len(v)) if isinstance(v,(dict,list)) else v
                failures.append(dict(case=i,difference=[diff[0],compact(diff[1]),compact(diff[2])]))
    return dict(match=not counts['mismatches'],cases=len(records),**counts,actions=dict(actions),kinds=dict(kinds),errors=dict(errors),first_failures=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=int,default=1000);p.add_argument('--seed',type=int,default=326741)
    p.add_argument('--assets',action='store_true',help='Use exported AnimationData assets and a separate golden fixture')
    p.add_argument('--reuse-unity',action='store_true');p.add_argument('--golden',action='store_true');args=p.parse_args()
    fixture=FIXTURE.with_name('animator-assets.jsonl.gz') if args.assets else FIXTURE
    if args.golden:
        with gzip.open(fixture,'rt') as stream:header=json.loads(next(stream));records=[json.loads(line) for line in stream]
        if header['schema']!=SCHEMA or header['cases']!=len(records):raise ValueError('Invalid animator fixture')
        report=check(records)
    else:
        if args.cases<1:p.error('--cases must be positive')
        directory=ROOT/('native/artifacts/animator-assets' if args.assets else 'native/artifacts/animator')
        inputs,assets=asset_queries() if args.assets else (list(queries(args.cases,args.seed)),None)
        records,hashes=reference(directory,inputs,args.reuse_unity)
        if assets is not None:hashes=dict(hashes,assets=assets)
        report=dict(check(records),provenance=hashes);(directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if report['match']:
            selected=set(range(min(32,len(records))))|set(random.Random(args.seed).sample(range(len(records)),min(128,len(records))))
            # Errors at arbitrary terminal sprite getter indices are repetitive;
            # preserve all early failures and sample those final getter failures.
            selected.update(i for i,r in enumerate(records) if isinstance(r['expected'],dict) and r['expected']['action']<len(r['query']['actions'])-2)
            if args.assets:selected=set(range(len(records)))
            header=dict(schema=SCHEMA,cases=len(selected),provenance=hashes,
                scope='Actual SpriteAnimator modes, mutable SpeedContainer references, frame/sprite/flip state and reentrant callback boundaries; opaque sprite IDs; no complete ability or spawn lifecycle')
            fixture.write_bytes(gzip.compress(encoded([header,*[records[i] for i in sorted(selected)]]).encode(),mtime=0))
    print(json.dumps(report,indent=2));return int(not report['match'])


if __name__=='__main__':raise SystemExit(main())
