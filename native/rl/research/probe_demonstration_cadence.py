"""Measure counterfactual effects of resampling expert inputs, without a model."""
import argparse
import json
from pathlib import Path
from replay_data import ReplayWorld, decode

VARIANTS={'human60':(1,0,True,True),'human15':(4,0,True,True),
          'human30':(2,0,True,True),'aim15':(4,0,True,False),
          'buttons15':(4,0,False,True),'human15phase3':(4,3,True,True)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays',type=Path,nargs='+')
    parser.add_argument('--player',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[];origin=240;index=args.player-1
    for path in args.replays:
        meta,inputs=decode(path)
        for name,(repeat,phase,aim,buttons) in VARIANTS.items():
            angles=inputs['angle'].copy();pressed=inputs['buttons'].copy()
            for t in range(origin+phase,meta['ticks']):
                source=origin+phase+(t-origin-phase)//repeat*repeat
                if aim: angles[t,index]=inputs[source,index]['angle']
                if buttons: pressed[t,index]=inputs[source,index]['buttons']
            out=args.output/(path.stem+'-'+name)
            with ReplayWorld(meta,out.with_suffix('.log')) as world:
                for t in range(meta['ticks']):
                    if world.ended: break
                    world.step(list(zip(angles[t],pressed[t])))
                state=world.state;p=state.players[index];telemetry=state.telemetry[index]
                result=dict(replay=path.name,replay_sha256=meta['sha256'],variant=name,player=args.player,
                    ticks=state.header.episode_ticks,arrows_fired=telemetry.arrows_fired,arrow_hits=telemetry.arrow_hits,
                    kills=p.kills,deaths=p.deaths)
                world.replay(out.with_suffix('.rep'))
            results.append(result);print(json.dumps(result),flush=True)
    (args.output/'result.json').write_text(json.dumps(dict(origin=origin,variants=VARIANTS,games=results),indent=2)+'\n')

if __name__=='__main__':main()
