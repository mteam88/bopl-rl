"""Does an unrestricted full-game policy retain the original four-second shot?"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from environment import Environment, configure
from league_evaluate import Controller
from policy import load
from prepare_league import digest
from train import atomic_json


def evaluate(checkpoint, out, trials=32):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(checkpoint).resolve()
    configure(out/'workers');torch.set_num_threads(1)
    model,metadata=load(checkpoint)
    if metadata.get('task')=='bow':raise ValueError('Full-game distribution required')
    rows=[]
    with Environment(seed=8211,limit=3600) as env:
        for trial in range(trials+1):
            seed=8111+trial;env.reset(seed,0)
            controller=Controller(dict(kind='policy',path=str(checkpoint)),seed,{str(checkpoint):model})
            state=model.initial(1,'cpu')
            argmax=trial==trials
            for step in range(60):
                if argmax:
                    with torch.no_grad():
                        logits,_,state=model.step(torch.from_numpy(env.obs[0].copy()).view(1,-1),state)
                    action=np.array([part.argmax().item() for part in logits[0].split([256,16])])
                else:action=controller.act(env.obs[0])
                actions=np.zeros((2,2),dtype=np.int64);actions[0]=action;env.step(actions)
                if env.state.telemetry[0].arrow_hits or env.state.players[0].deaths or env.ended:break
            row=dict(seed=seed,argmax=argmax,success=bool(env.state.telemetry[0].arrow_hits),
                     shots=int(env.state.telemetry[0].arrows_fired),deaths=int(env.state.players[0].deaths),ticks=4*(step+1))
            if trial in (0,trials):env.replay(out/('argmax.rep' if argmax else 'sample.rep'))
            rows.append(row)
    result=dict(schema=1,checkpoint_sha256=digest(checkpoint),map=0,side=0,world_limit=3600,
                trial_limit=240,controls='full, unrestricted',opponent='idle',
                successes=sum(r['success'] for r in rows[:-1]),trials=trials,argmax=rows[-1]['success'],games=rows)
    atomic_json(out/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text())
    for run in protocol['runs']:
        path=out/'retention'/run['name']
        if (path/'result.json').exists():continue
        result=json.loads((out/'runs'/run['name']/'result.json').read_text())
        checkpoint=out/'runs'/run['name']/'checkpoints'/Path(result['checkpoint']).name
        r=evaluate(checkpoint,path)
        print(run['name'],r['successes'],r['trials'],r['argmax'],flush=True)


if __name__=='__main__':main()
