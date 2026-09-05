"""Original DestroyFix, component cleanup and physics removal across actor ticks."""
import argparse
import copy
import json
from actor_stream import ROOT, capture, verify, promote, golden
from verify_actor_entity import queries as entity_queries
from verify_actors import ONE, DT


def queries(seed):
    for case, base in enumerate(entity_queries(6, seed, 20, False)):
        for phase in ['commands', 'init', 'update', 'late', 'physics', 'rope_mesh', 'duplicate', 'missing', 'children']:
            q = copy.deepcopy(base)
            q['lifecycle_case'] = phase
            r = q['runtime']; r['scheduler']['loaded'] = False
            q['scripts'] = {}
            for actor in r['actors']:
                actor['entity'].update(present=True, destroyed=False)
                actor['slime']['managed_destroyed'] = False
            for action in q['actions']:
                action.update(commands=[], enabled=[True]*len(r['components']), scale=ONE, dt=DT)
            victim = case
            if phase == 'children':
                victim=0
                for i in range(1,len(r['actors'])):
                    r['scheduler']['nodes'][i]['parent']=0
                    r['scheduler']['nodes'][0]['children'].append(i)
            command = dict(op='destroy_actor', actor=victim)
            if phase in ['commands', 'duplicate', 'missing', 'children']:
                q['actions'][5]['commands'] = [command]
                if phase == 'duplicate': q['actions'][5]['commands'].append(command)
                if phase == 'missing': q['actions'][6]['commands'] = [command]
            else:
                key = f'{phase}:*'
                if phase in ['init', 'update', 'late']:
                    component = next(i for i, c in enumerate(r['components']) if c['actor'] == victim)
                    key = f'{phase}:{component}'
                q['scripts'][key] = ([[]]*5 if phase != 'init' else []) + [[command]]
            yield q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=781436)
    p.add_argument('--reuse-unity',action='store_true')
    p.add_argument('--golden',action='store_true')
    args=p.parse_args(); fixture=ROOT/'native/fixtures/actor-lifecycle.jsonl.gz'
    if args.golden: result=golden(fixture)
    else:
        directory=ROOT/'native/artifacts/actor-lifecycle'
        inputs, hashes=capture(directory,list(queries(args.seed)),args.reuse_unity)
        result=dict(verify(directory,inputs),provenance=hashes)
        (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['match']: promote(directory,fixture,inputs,set(range(len(inputs))),hashes,__doc__)
    print(json.dumps(result,indent=2));return int(not result['match'])

if __name__=='__main__':raise SystemExit(main())
