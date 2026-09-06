"""Minimal game-control baseline: point toward the opponent, hold Bow, release.

Uses only delayed policy features and a local counter. No human target actions,
raw current game state, ballistic model, platform planning or trained weights.
Scoped to the two-player demonstrations with Bow in player 2's first slot.
"""
import argparse
import json
from pathlib import Path
import torch
from evaluate_imitation import play
from replay_data import decode

class BowBaseline:
    def __init__(self,repeat=2): self.repeat=repeat
    def initial(self,agents,device): return torch.zeros(1,agents,1,device=device)
    def step(self,obs,state):
        counter=int(state[0,0,0]);buttons=0;angle=0
        if counter==0 and obs[0,10+16+42]>.5 and obs[0,10+148+3]>.5:
            counter=64//self.repeat
        if counter>0:
            dx=obs[0,10+148+9];dy=obs[0,10+148+10]
            angle=int(torch.round(torch.remainder(torch.atan2(dy,dx),2*torch.pi)*255/(2*torch.pi)))%255+1
            buttons=2;counter-=1
            if counter==0:counter=-16//self.repeat
        elif counter<0:counter+=1
        logits=torch.full((1,272),-1000.);logits[0,angle]=0;logits[0,256+buttons]=0
        return logits,torch.zeros(1),torch.tensor([[[float(counter)]]])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    for path in args.replays:
        meta,_=decode(path)
        if len(meta['players'])!=2 or meta['players'][1]['abilities'][0]!=4:
            raise ValueError('Baseline requires two players and Bow in player 2 slot 1')
    args.output.mkdir(parents=True,exist_ok=False)
    results=[play(path,args.output/path.stem,2,'point-hold-release',BowBaseline(),2) for path in args.replays]
    (args.output/'result.json').write_text(json.dumps(dict(hold_ticks=64,release_gap_ticks=16,repeat=2,games=results),indent=2)+'\n')
