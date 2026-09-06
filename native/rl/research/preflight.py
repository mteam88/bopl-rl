"""Meaningful checks for reaction delay, recurrent resets, returns, and reload."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from environment import Environment,ROOT,configure,OBS_SIZE
from policy import Policy,load
from train import advantages

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(67)
    r=torch.tensor([[1.],[2.],[3.]])
    a=advantages(r,torch.zeros_like(r),torch.tensor([[0.],[1.],[0.]]),torch.tensor([10.]),gamma=.9,lam=1)
    torch.testing.assert_close(a,torch.tensor([[2.8],[2.],[12.]]))
    results={'return_terminal_and_bootstrap':'pass'}
    for device in ('cpu','cuda'):
        policy=Policy(OBS_SIZE).to(device)
        obs=torch.randn(8,2,OBS_SIZE,device=device)*.1
        reset=torch.zeros(8,2,dtype=torch.bool,device=device);reset[3,0]=True;reset[6,1]=True
        initial=torch.randn(1,2,128,device=device)*.1; state=initial.clone(); expected=[]
        with torch.no_grad():
            for t in range(8):
                state=state*(~reset[t]).view(1,-1,1)
                logits,_,state=policy.step(obs[t],state);expected.append(logits)
            actual,_=policy.sequence(obs,initial,reset)
            torch.testing.assert_close(actual,torch.stack(expected),rtol=1e-4,atol=1e-5)
            file=args.output/(device+'-reload.pt')
            torch.save(dict(obs_size=OBS_SIZE,hidden=128,policy=policy.state_dict()),file)
            reloaded,_=load(file,device);again,_=reloaded.sequence(obs,initial,reset)
            torch.testing.assert_close(actual,again,rtol=0,atol=0)
        results[device+'_recurrent_reset_and_reload']='pass'
    configure(args.output/'workers')
    with Environment(seed=18,limit=120) as env:
        initial=env.obs.copy()
        for i in range(8):
            env.step([[64,4],[192,4]])
            tick=env.state.header.episode_ticks
            assert round(float(env.obs[0,2])*120)==max(0,tick-9)
            assert np.isfinite(env.obs).all()
            np.testing.assert_array_equal(env.obs[:,-4:],[[0,0,1,0],[0,0,1,0]])
        env.reset(18)
        # The constructor mixes its seed once; reset itself must be repeatable.
        again=env.obs.copy();env.step([[0,15],[0,15]]);env.reset(18)
        np.testing.assert_array_equal(env.obs,again)
        assert all(s['abilities']==[2,4,14] for s in env.frame()['slimes'])
    results['real_game_delay_reset_loadout']='pass'
    (args.output/'result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))

if __name__=='__main__': main()
