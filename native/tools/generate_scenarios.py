"""Generate reproducible held-input trajectories; game RNG remains the pinned reset seed."""
import argparse,json,random
from pathlib import Path

def generate(seed,ticks):
    rng=random.Random(seed);frames=[[[0,0],[0,0]] for _ in range(240)]
    held=[[0,0],[0,0]];remaining=[0,0]
    for tick in range(ticks):
        for player in range(2):
            if remaining[player]==0:
                held[player]=[rng.randrange(256),rng.choices([0,1,2,3],[5,3,1,1])[0]]
                remaining[player]=rng.randrange(1,31)
            remaining[player]-=1
        frames.append([a.copy() for a in held])
    return dict(setup=dict(seed=12345,level=0),action_seed=seed,frames=frames)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--seed',type=int,default=1);p.add_argument('--ticks',type=int,default=360);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(generate(args.seed,args.ticks),separators=(',',':'))+'\n')
