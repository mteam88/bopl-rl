"""Counterfactual replay evaluation: replace one human with a policy.

Other players retain recorded inputs; these are demonstrations against a fixed
input stream, not a competitive win-rate benchmark. Compare full-rate human,
15 Hz human, and policies on identical initial conditions and tick budgets.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import torch
from policy import load
from replay_data import ReplayWorld, decode, previous, hold_lengths

def play(path,out,player,mode,model=None):
    meta,inputs=decode(path); index=player-1; origin=240
    history=deque(maxlen=10); actions=[]; frames=[]
    memory=model.initial(1,'cpu') if model is not None else None
    prior=[(0,0)]*len(meta['players']); held=None
    with ReplayWorld(meta,out.with_suffix('.log')) as world:
        for t in range(meta['ticks']):
            if world.ended: break
            if t >= origin:
                history.append(world.pack())
                if (t-origin)%4==0:
                    if mode=='human60': chosen=(int(inputs[t,index]['angle']),int(inputs[t,index]['buttons']))
                    elif mode=='human15': chosen=(int(inputs[t,index]['angle']),int(inputs[t,index]['buttons']))
                    else:
                        obs=history[0].copy();previous(obs,prior)
                        with torch.no_grad():
                            logits,_,memory=model.step(torch.from_numpy(obs[index:index+1]),memory)
                        chosen=tuple(int(p.argmax(-1).item()) for p in logits.split([256,16],dim=-1))
                    held=chosen;actions.append(chosen)
                    state=world.state; p=state.players[index]
                    frames.append(dict(t=t,action=chosen,x=p.x/4294967296.,y=p.y/4294967296.,stocks=p.stocks,
                        arrows=state.telemetry[index].arrows_fired,hits=state.telemetry[index].arrow_hits))
            current=list(zip(inputs[t]['angle'].astype(int),inputs[t]['buttons'].astype(int)))
            if t>=origin and mode!='human60': current[index]=held
            world.step(current)
            if t>=origin and (t-origin)%4==3:
                # Exactly the last four-tick policy decision; do not substitute
                # any future human target when the model drives itself.
                prior=current.copy()
        state=world.state;telemetry=state.telemetry[index];p=state.players[index]
        result=dict(replay=meta['path'],replay_sha256=meta['sha256'],player=player,mode=mode,
            ticks=state.header.episode_ticks,requested_ticks=meta['ticks'],arrows_fired=telemetry.arrows_fired,
            arrow_hits=telemetry.arrow_hits,kills=p.kills,deaths=p.deaths,stocks=p.stocks)
        bow=meta['players'][index]['abilities'].index(4)
        world.replay(out.with_suffix('.rep'))
        _,executed=decode(out.with_suffix('.rep'))
        result['completed_bow_holds_ticks']=hold_lengths(executed[:,index]['buttons'],bow+1,origin)
    out.with_suffix('.json').write_text(json.dumps(dict(result=result,frames=frames),indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays',nargs='+',type=Path)
    parser.add_argument('--player',type=int,required=True)
    parser.add_argument('--checkpoint',action='append',default=[],help='label=/absolute/checkpoint.pt')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    models={}; identities={}
    for value in args.checkpoint:
        label,path=value.split('=',1)
        if not label.replace('-','').isalnum() or label in ('human60','human15') or label in models:
            raise ValueError('Invalid or duplicate policy label')
        models[label]=load(Path(path))[0].eval()
        identities[label]=dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    results=[]
    for path in args.replays:
        for mode,model in [('human60',None),('human15',None),*models.items()]:
            results.append(play(path,args.output/(path.stem+'-'+mode),args.player,mode,model))
    (args.output/'result.json').write_text(json.dumps(dict(checkpoints=identities,games=results),indent=2)+'\n')

if __name__=='__main__': main()
