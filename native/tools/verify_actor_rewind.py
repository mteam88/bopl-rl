"""Original RewindRecorder history writes, full ring contents and lifecycle gates."""
import argparse,copy,json
from collections import Counter
from actor_stream import ROOT,capture,verify,promote,golden
from verify_actor_spawn import queries as spawn_queries
from verify_actors import ONE,vec

def queries(seed):
    for base in spawn_queries(seed):
        if base['spawn_case']!='init':continue
        for mode in ['record','time_stop','gates','respawn','destroy','controller_destroyed','timers_short','buffer_short']:
            q=copy.deepcopy(base);q['rewind_case']=mode;r=q['runtime'];victim=len(r['actors'])-1
            for i,actor in enumerate(r['actors']):
                sample=dict(position=vec(),velocity=vec(),external_velocity=vec(),rotation=0,cooldowns=[0]*3)
                actor['rewind']=dict(ring=[copy.deepcopy(sample) for _ in range(188)],head=0,count=0,rewinding=False)
                cid=len(r['components']);r['components'].append(dict(actor=i,role='rewind'))
                r['scheduler']['components'].append(dict(owner=i,kind='updatable',present=True,destroyed=False,hierarchy=8,enabled=True,ignore_activity=False))
                r['scheduler']['live'].append(cid)
            while len(q['actions'])<410:q['actions'].append(copy.deepcopy(q['actions'][-1]))
            for tick,action in enumerate(q['actions']):
                action['enabled']=[True]*len(r['components'])
                for i,inp in enumerate(action['inputs']):inp.update(direction=vec(1 if (tick//35+i)%2 else -1),jump=tick%71==0)
                if mode=='time_stop':action['scale']=0 if tick%80<40 else ONE
                if mode=='gates':
                    if tick%40==0:action['commands'].append(dict(op='body_simulated',actor=victim,value=(tick//40)%2==0))
                    if tick%47==0:action['commands'].append(dict(op='actor_active',actor=victim,value=(tick//47)%2==0))
                    action['enabled'][-1]=tick%11!=0
            if mode=='time_stop':r['scheduler']['players'][0]['protected']=True
            if mode=='respawn':
                for tick,op in [(51,'prepare'),(71,'fast'),(230,'prepare'),(260,'respawn')]:
                    q['actions'][tick]['commands']=[dict(op='actor_spawn',actor=victim,action=op,position=vec(3,12))]
            if mode=='destroy':q['actions'][205]['commands']=[dict(op='destroy_actor',actor=victim)]
            if mode=='controller_destroyed':
                for tick,value in [(25,True),(85,False)]:q['actions'][tick]['commands']=[dict(op='controller_destroyed',actor=victim,value=value)]
            if mode=='timers_short':
                # Slime.Init replaces its timer array; omit spawn initialization
                # so the original recorder observes the deliberately short array.
                del r['actors'][victim]['slime']['spawn']
                r['actors'][victim]['slime']['abilities']['timers']=[0,0]
                for action in q['actions']:
                    for i,c in enumerate(r['components']):action['enabled'][i]=c['role']=='rewind'
            if mode=='buffer_short':r['actors'][victim]['rewind']['ring']=[]
            yield q

def coverage(directory,qs):
    counts=Counter();previous={}
    with (directory/'unity.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);state=row.get('state',row.get('result',{}).get('failed'))
            if state is None:continue
            case=row['case'];before=previous.get(case,qs[case]['runtime'])
            for a,b in zip(before['actors'],state['actors']):
                old,new=a['rewind'],b['rewind'];written=old['head']!=new['head']
                counts['writes']+=written;counts['wraps']+=written and new['head']==0
                counts['full_ring_frames']+=new['count']==188
                counts['writes_while_body_unsimulated']+=written and not b['entity']['body_simulated']
                action=qs[case]['actions'][row.get('action',row.get('result',{}).get('action',0))]
                counts['writes_at_zero_time_scale']+=written and action['scale']==0
                counts['preserved_frames']+=not written and old==new
            previous[case]=state
    return dict(counts)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=725606);p.add_argument('--golden',action='store_true');p.add_argument('--reuse-unity',action='store_true');p.add_argument('--directory');p.add_argument('--cases',type=int,default=0);a=p.parse_args()
    name='actor-rewind';fixture=ROOT/f'native/fixtures/{name}.jsonl.xz'
    if a.golden:result=golden(fixture)
    else:
        directory=ROOT/f'native/artifacts/{a.directory or name}';qs=list(queries(a.seed));qs=qs[:a.cases] if a.cases else qs
        inputs,hashes=capture(directory,qs,a.reuse_unity);result=dict(verify(directory,inputs),provenance=hashes)
        if result['match']:result['rewind_coverage']=coverage(directory,inputs)
        (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match'] and not a.cases:promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'])
if __name__=='__main__':raise SystemExit(main())
