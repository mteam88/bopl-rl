"""Check real-game reaction delay, observation finiteness, resets and loadouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from environment import Environment,ROOT,configure,OBS_SIZE
from layout import HOLD_MAX,OBS_PREVIOUS_OFFSET

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--game',type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    results={}
    configure(args.output/'workers', args.game)
    with Environment(seed=18,limit=120) as env:
        for i in range(8):
            env.step([[64,4,0],[192,4,0]])  # aim, buttons (arrow), hold index 0 = one decision
            tick=env.state.header.episode_ticks
            assert round(float(env.obs[0,2])*120)==max(0,tick-9)
            assert np.isfinite(env.obs).all()
            tail=env.obs[:,OBS_PREVIOUS_OFFSET:]
            np.testing.assert_array_equal(tail[:,3:7],[[0,0,1,0],[0,0,1,0]])  # applied buttons
            np.testing.assert_array_equal(tail[:,7],0)  # no latch remaining with hold length 1
            np.testing.assert_array_equal(tail[:,8:12],[[0,0,(i+1)/HOLD_MAX,0]]*2)  # arrow held i+1 consecutive decisions
        env.reset(18)
        # The constructor mixes its seed once; reset itself must be repeatable.
        again=env.obs.copy();env.step([[0,15,0],[0,15,0]]);env.reset(18)
        np.testing.assert_array_equal(env.obs,again)
        assert all(s['abilities']==[2,4,14] for s in env.frame()['slimes'])
    results['real_game_delay_reset_loadout']='pass'
    (args.output/'result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))

if __name__=='__main__': main()
