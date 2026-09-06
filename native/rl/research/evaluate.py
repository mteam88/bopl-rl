"""Fixed-seed, side-swapped evaluation with downloadable native and visual replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from environment import Environment,configure
from policy import load
from train import atomic_json
from telemetry import FIELDS,summarize

def sample(logits,generator):
    return torch.stack([torch.multinomial(part.softmax(-1),1,generator=generator).squeeze(-1)
                        for part in logits.split([256,16],-1)],-1)

def wilson(wins,n):
    if not n: return [0,1]
    z=1.96; p=wins/n; denom=1+z*z/n
    middle=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0,middle-half),min(1,middle+half)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--games-per-opponent',type=int,default=8)
    parser.add_argument('--heldout-games',type=int,default=4)
    parser.add_argument('--seed-base',type=int,default=10001)
    parser.add_argument('--heldout-seed-base',type=int,default=20001)
    parser.add_argument('--selfplay-seed',type=int,default=30001)
    parser.add_argument('--suite',choices=('diagnostic','final'),default='diagnostic')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if min(args.games_per_opponent,args.heldout_games,args.workers)<1:
        parser.error('Positive game and worker counts required')
    if args.games_per_opponent%2 or args.heldout_games%2:
        parser.error('An even number of games is required for side-swapped seed pairs')
    torch.set_num_threads(1)
    policy,checkpoint=load(args.checkpoint)
    initial,_=load(args.initial)
    step=checkpoint['step']; config=checkpoint['config']
    out=args.run/('evaluations' if args.suite=='diagnostic' else 'final-evaluations')/f'{step:09d}'
    if (out/'result.json').exists():
        print(f'Evaluation {step} already complete',flush=True); return
    out.mkdir(parents=True,exist_ok=True)
    configure(out/'workers')
    tasks=[]
    # Identical seed/side pairs at every checkpoint. Held-out maps are never
    # added to the training stream by evaluation.
    for opponent in ('random','initial'):
        for game in range(args.games_per_opponent):
            tasks.append(dict(opponent=opponent,map=0,seed=args.seed_base+game//2,side=game%2,
                              save=game==0,name=f'{opponent}-{game}'))
    for map_id in (1,2):
        for game in range(args.heldout_games):
            tasks.append(dict(opponent='random',map=map_id,seed=args.heldout_seed_base+game//2,side=game%2,
                              save=game==0,name=f'map-{map_id}-{game}'))
    tasks.append(dict(opponent='selfplay',map=0,seed=args.selfplay_seed,side=0,save=True,name='selfplay'))
    started=time.monotonic()
    # Each thread owns its game and its sampling generator; the model weights
    # are immutable. Reuse each game process across tasks to avoid startup cost.
    buckets=[tasks[i::args.workers] for i in range(min(args.workers,len(tasks)))]
    def work(bucket):
        results=[]
        with Environment(seed=90001+tasks.index(bucket[0]),limit=config['episode_ticks']) as env:
            for task in bucket:
                env.reset(task['seed'],task['map'])
                state=policy.initial(2,'cpu'); other_state=initial.initial(2,'cpu')
                generator=torch.Generator().manual_seed(task['seed']*7+task['side'])
                rng=np.random.default_rng(task['seed']*11+task['side'])
                frames=[env.frame()] if task['save'] else []
                button_counts=np.zeros(4,dtype=np.int64); decisions=0; events=[]
                while not env.ended:
                    observation=torch.from_numpy(env.obs.copy())
                    with torch.no_grad():
                        logits,_,state=policy.step(observation,state)
                        actions=sample(logits,generator)
                        other=1-task['side']
                        if task['opponent']=='initial':
                            old_logits,_,other_state=initial.step(observation,other_state)
                            actions[other]=sample(old_logits,generator)[other]
                        elif task['opponent']=='random':
                            actions[other]=torch.tensor([rng.integers(256),rng.integers(16)])
                    env.step(actions.numpy())
                    events.append(env.telemetry[task['side']].copy())
                    buttons=int(actions[task['side'],1])
                    for i in range(4): button_counts[i]+=bool(buttons&(1<<i))
                    decisions+=1
                    if task['save']:
                        frame=env.frame(); frame['actions']=actions.tolist(); frames.append(frame)
                final=env.frame(); me=final['players'][task['side']]
                timeout=bool(final['flags']&2); win=bool(me['flags']&2) and not timeout
                result={**task,'win':win,'timeout':timeout,'kills':me['kills'],'deaths':me['deaths'],
                        'ticks':final['t'],'behavior':summarize(np.asarray(events)),
                        'action_rates':dict(zip(['jump','dash','arrow','platform'],(button_counts/max(1,decisions)).tolist()))}
                if task['save']:
                    replay=out/(task['name']+'.rep'); env.replay(replay)
                    visual=out/(task['name']+'.json')
                    atomic_json(visual,dict(schema=1,step=step,task=task,result=result,frames=frames,
                                           rendering='physics state replay',tick_rate=60,decision_rate=15))
                    result.update(replay=replay.name,visual=visual.name,
                                  replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
                results.append(result)
                print(json.dumps(dict(step=step,**result)),flush=True)
        return results
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        results=[result for bucket in workers.map(work,buckets) for result in bucket]
    groups=[]
    for opponent,map_id in [('random',0),('initial',0),('random',1),('random',2)]:
        selected=[r for r in results if r['opponent']==opponent and r['map']==map_id]
        wins=sum(r['win'] for r in selected); n=len(selected)
        groups.append(dict(opponent=opponent,map=map_id,games=n,wins=wins,win_rate=wins/n,
                           interval=wilson(wins,n),timeouts=sum(r['timeout'] for r in selected),
                           seed_pairs=n//2,
                           behavior_totals={key:sum(r['behavior']['totals'][key] for r in selected) for key in FIELDS},
                           mean_kills=float(np.mean([r['kills'] for r in selected])),
                           mean_deaths=float(np.mean([r['deaths'] for r in selected])),
                           mean_duration=float(np.mean([r['ticks']/60 for r in selected]))))
    report=dict(schema=1,step=step,suite=args.suite,checkpoint=args.checkpoint.name,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                baseline_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
                seed_base=args.seed_base,heldout_seed_base=args.heldout_seed_base,selfplay_seed=args.selfplay_seed,
                interval_note='Descriptive Wilson interval; does not adjust for correlation within seed pairs',
                elapsed=time.monotonic()-started,groups=groups,games=results)
    atomic_json(out/'result.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
