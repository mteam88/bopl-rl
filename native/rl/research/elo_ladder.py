"""Round-robin ELO ladder over a run's checkpoints plus fixed anchors.

Plays side-swapped fixed-seed pools among evenly spaced native checkpoints
(policy-%09d.bin; legacy .pt files are display-only and never loaded), plus
anchors: the random policy, optional scripted bots, and optional frozen native
reference checkpoints. Ratings come from a Bradley-Terry maximum-likelihood
fit (timeouts count as draws) anchored at random = 0 Elo, so numbers are
comparable across reruns of the same pool. Output is written atomically to
<run>/ladder/ladder.json, which the dashboard reads as-is.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from environment import Environment,configure
from evaluation_protocol import map_ids,opponents

SCHEMA=1
ELO_SCALE=400/math.log(10)


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload)+'\n'); temporary.replace(path)


class Model:
    """A native checkpoint participant (research/native_policy.py contract)."""
    def __init__(self,name,path,kind,step=None,library=None):
        from native_policy import load
        self.name=name; self.kind=kind; self.step=step; self.path=Path(path)
        self.policy,self.meta=load(self.path,**(dict(library=library) if library else {}))
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()

    def start(self,task,side,rng):
        import torch
        state=self.policy.initial(2,'cpu')
        def act(observation):
            nonlocal state
            with torch.no_grad():
                batch=torch.from_numpy(np.ascontiguousarray(observation,dtype=np.float32))
                logits,_,state=self.policy.step(batch,state)
            row=as_numpy(logits)[side]
            action=[]
            for part in (row[:256],row[256:272]):
                weights=np.exp(part-part.max()); weights/=weights.sum()
                action.append(rng.choice(len(part),p=weights))
            return np.asarray(action,dtype=np.int64)
        return act


class Random:
    name='random'; kind='anchor'; step=None; sha256=None

    def start(self,task,side,rng):
        return lambda observation: np.asarray([rng.integers(256),rng.integers(16)],dtype=np.int64)


class Scripted:
    kind='anchor'; step=None

    def __init__(self,spec):
        from scripted_opponents import identity
        self.spec=spec; self.name=spec.name; self.sha256=identity(spec)

    def start(self,task,side,rng):
        from scripted_opponents import ScriptedOpponent
        bot=ScriptedOpponent(self.spec,int(rng.integers(2**31)))
        return lambda observation: np.asarray(bot.act(as_numpy(observation)[side]),dtype=np.int64)


def select_checkpoints(run,count):
    """Evenly spaced playable checkpoints, always keeping first and last."""
    rows=[]
    for path in sorted(run.glob('checkpoints/policy-*.bin')):
        match=re.fullmatch(r'policy-(\d+)\.bin',path.name)
        if match: rows.append((int(match.group(1)),path))
    if len(rows)<=count: return rows
    picks=sorted({round(i*(len(rows)-1)/(count-1)) for i in range(count)})
    return [rows[i] for i in picks]


def bradley_terry(names,records,anchor):
    """MLE ratings in Elo units with the anchor fixed at 0.

    records: {(a,b): [wins_a, wins_b, draws]} with a,b participant names.
    One virtual draw per played pair regularizes undefeated participants.
    """
    index={name:i for i,name in enumerate(names)}
    ratings=np.zeros(len(names))
    pairs=[(index[a],index[b],wa+d/2+.5,wb+d/2+.5) for (a,b),(wa,wb,d) in records.items()]
    for _ in range(500):
        gradient=np.zeros_like(ratings)
        for a,b,score_a,score_b in pairs:
            expected=1/(1+math.exp(ratings[b]-ratings[a]))
            gradient[a]+=score_a-(score_a+score_b)*expected
            gradient[b]+=score_b-(score_b+score_a)*(1-expected)
        ratings+=gradient*(.5/max(1,len(pairs)))
        ratings-=ratings[index[anchor]]
        if np.abs(gradient).max()<1e-9: break
    return {name:ratings[i]*ELO_SCALE for name,i in index.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoints',type=int,default=8,help='Pool size sampled evenly across the run')
    parser.add_argument('--games-per-pair',type=int,default=8,help='Per map; even, half per side')
    parser.add_argument('--maps',type=map_ids,default=None,help='Defaults to the run config training maps')
    parser.add_argument('--anchors',type=Path,help='Manifest of frozen native reference checkpoints (evaluation opponents format)')
    parser.add_argument('--scripts',type=Path,help='Scripted bot manifest; each bot becomes an anchor. '
                        'Depends on the legacy scripted_opponents module, which is slated for deletion '
                        'in the native cutover — unavailable until a native reimplementation exists')
    parser.add_argument('--seed-base',type=int,default=50001)
    parser.add_argument('--episode-ticks',type=int,default=None,help='Defaults to the run config value')
    parser.add_argument('--library',type=Path,default=None,help='CPU evaluation library for native_policy.load')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.games_per_pair%2 or args.games_per_pair<2:
        parser.error('An even positive games count is required for side-swapped seed pairs')
    config=json.loads((args.run/'config.json').read_text())
    maps=args.maps if args.maps is not None else config['training_maps']
    limit=args.episode_ticks or config['episode_ticks']
    participants=[Random()]
    if args.scripts:
        from scripted_opponents import load_specs
        participants+=[Scripted(spec) for spec in load_specs(args.scripts)]
    for item in opponents(args.anchors) if args.anchors else []:
        participants.append(Model(item['name'],item['path'],'anchor',library=args.library))
    for step,path in select_checkpoints(args.run,args.checkpoints):
        participants.append(Model(f'checkpoint-{step:09d}',path,'checkpoint',step,library=args.library))
    if sum(p.kind=='checkpoint' for p in participants)<2:
        raise SystemExit('Need at least two native .bin checkpoints to rate')
    by_name={p.name:p for p in participants}
    tasks=[]
    names=sorted(by_name)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            for map_id in maps:
                for pair in range(args.games_per_pair//2):
                    seed=args.seed_base+pair
                    tasks.append(dict(a=a,b=b,map=map_id,seed=seed,side=0))
                    tasks.append(dict(a=a,b=b,map=map_id,seed=seed,side=1))
    out=args.run/'ladder'; out.mkdir(parents=True,exist_ok=True)
    configure(out/'workers')
    started=time.monotonic()

    def play(bucket):
        results=[]
        with Environment(seed=args.seed_base+len(tasks),limit=limit) as env:
            for task in bucket:
                env.reset(task['seed'],task['map'])
                rng=np.random.default_rng((task['seed'],task['map'],task['side'],
                                           int(hashlib.sha256((task['a']+task['b']).encode()).hexdigest()[:8],16)))
                actors={task['side']:by_name[task['a']].start(task,task['side'],rng),
                        1-task['side']:by_name[task['b']].start(task,1-task['side'],rng)}
                while not env.ended:
                    observation=env.obs.copy()
                    env.step(np.stack([actors[0](observation),actors[1](observation)]))
                final=env.frame()
                timeout=bool(final['flags']&2)
                a_won=bool(final['players'][task['side']]['flags']&2) and not timeout
                b_won=bool(final['players'][1-task['side']]['flags']&2) and not timeout
                results.append(dict(task,ticks=final['t'],
                                    outcome='draw' if timeout or a_won==b_won else ('a' if a_won else 'b')))
        return results

    buckets=[tasks[i::args.workers] for i in range(min(args.workers,len(tasks)))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        games=[game for bucket in pool.map(play,buckets) for game in bucket]
    records={}
    for game in games:
        row=records.setdefault((game['a'],game['b']),[0,0,0])
        row[{'a':0,'b':1,'draw':2}[game['outcome']]]+=1
    ratings=bradley_terry(names,records,'random')
    counts={name:[0,0,0] for name in names}  # wins, losses, draws
    for (a,b),(wa,wb,d) in records.items():
        counts[a][0]+=wa; counts[a][1]+=wb; counts[a][2]+=d
        counts[b][0]+=wb; counts[b][1]+=wa; counts[b][2]+=d
    atomic_json(out/'ladder.json',dict(
        schema=SCHEMA,updated=time.time(),elapsed=time.monotonic()-started,
        games=len(games),games_per_pair=args.games_per_pair,maps=maps,
        seed_base=args.seed_base,episode_ticks=limit,
        note='Bradley-Terry MLE, timeouts as draws, one virtual draw per pair, random anchored at 0 Elo',
        ratings=[dict(name=name,kind=by_name[name].kind,step=by_name[name].step,
                      sha256=by_name[name].sha256,rating=round(ratings[name],1),
                      games=sum(counts[name]),wins=counts[name][0],
                      losses=counts[name][1],draws=counts[name][2]) for name in names],
        matches=[dict(a=a,b=b,wins_a=wa,wins_b=wb,draws=d) for (a,b),(wa,wb,d) in sorted(records.items())]))
    for name in sorted(names,key=lambda n:-ratings[n]):
        print(f'{ratings[name]:8.1f}  {by_name[name].kind:10s}  {name}',flush=True)


if __name__=='__main__': main()
