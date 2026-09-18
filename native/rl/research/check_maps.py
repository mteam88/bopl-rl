"""Qualify complete games and exact delayed observations on declared maps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from environment import Environment,configure
from evaluation_protocol import map_ids
import map_catalogue
from layout import ACT_AIM,ACT_BUTTONS
from telemetry import FIELDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maps',type=map_ids,default=[0,1,2,3,4])
    parser.add_argument('--run',type=Path,help='interpret numeric --maps through this legacy run catalogue')
    parser.add_argument('--games',type=int,default=8)
    args=parser.parse_args()
    if args.run:
        args.maps=[map_catalogue.to_identity(args.run,map_id) for map_id in args.maps]
    args.output.mkdir(parents=True,exist_ok=False)
    configure(args.output/'workers');results=[]
    with Environment(limit=3600) as env:
        for map_id in args.maps:
            for game in range(args.games):
                seed=91001+game;env.reset(seed,map_id)
                initial=env.obs.copy()
                env.step([[64,4,0],[192,4,0]]);env.reset(seed,map_id)
                np.testing.assert_array_equal(env.obs,initial)
                rng=np.random.default_rng(seed);ticks=0;decisions=0;digest=hashlib.sha256()
                while not env.ended:
                    # Random aim and buttons; hold index 0 (length 1) so no decision is latched.
                    actions=np.stack([rng.integers(ACT_AIM,size=2),rng.integers(ACT_BUTTONS,size=2),np.zeros(2,np.int64)],axis=-1)
                    env.step(actions);decisions+=1
                    assert np.isfinite(env.obs).all()
                    assert not env.obs[:,[5,7]].any(), 'Policy observation capacity exceeded'
                    ticks+=int(env.telemetry[0,FIELDS.index('ticks')])
                    now=env.state.header.episode_ticks
                    if not env.ended:
                        assert round(float(env.obs[0,2])*3600)==max(0,now-9)
                    digest.update(env.obs.tobytes())
                assert ticks==env.state.header.episode_ticks and ticks>0
                assert env.telemetry[:,FIELDS.index('episodes')].sum()==2
                replay=args.output/f'map-{map_id}-game-{game}.rep';env.replay(replay)
                results.append(dict(map=map_id,seed=seed,ticks=ticks,decisions=decisions,
                                    observations_sha256=digest.hexdigest(),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest()))
                print(json.dumps(results[-1]),flush=True)
    (args.output/'result.json').write_text(json.dumps(dict(status='pass',map_catalogue=map_catalogue.IDENTITY,
                                                           maps=args.maps,games=results),indent=2)+'\n')


if __name__=='__main__': main()
