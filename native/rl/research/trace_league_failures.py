"""Reproduce fixed old evaluation cases and save actual observations and actions."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
from train import ROOT, atomic_json
from continuation import digest
from environment import Environment, configure
from league_evaluate import Controller
from policy import load
from telemetry import FIELDS,summarize


def trace(case,out):
    original=ROOT/case['original'];assert digest(original)==case['sha256']
    expected=json.loads(original.read_text());directory=out/(case['run']+'-'+case['case'])
    if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text())
    directory.mkdir(parents=True,exist_ok=False);configure(directory/'workers')
    checkpoint=(ROOT/case['checkpoint']) if 'checkpoint' in case else ROOT/'native/artifacts/rl/opponent-league/runs'/case['run']/'checkpoints'/Path(expected['learner']['path']).name
    assert digest(checkpoint)==expected['learner']['sha256']
    model,_=load(checkpoint);side=expected['side'];seed=expected['seed']
    descriptions=[None,None]
    descriptions[side]={**expected['learner'],'path':str(checkpoint)};descriptions[1-side]=expected['opponent']
    controllers=[None,None];controllers[side]=Controller(descriptions[side],seed*7+side,{str(checkpoint):model})
    controllers[1-side]=Controller(descriptions[1-side],seed*11+1-side,{})
    obs=[];actions=[];events=[];frames=[]
    with Environment(seed=90001,map_id=expected['map'],limit=3600) as env:
        env.reset(seed,expected['map']);initial=env.frame();frames.append(initial)
        while not env.ended:
            obs.append(env.obs.copy())
            action=np.stack([controllers[i].act(env.obs[i]) for i in range(2)])
            actions.append(action);env.step(action);events.append(env.telemetry.copy());frames.append(env.frame())
        final=env.frame();env.replay(directory/'game.rep')
    actual_behavior=summarize(np.asarray(events)[:,side])
    assert actual_behavior==expected['behavior']
    assert summarize(np.asarray(events)[:,1-side])==expected['opponent_behavior']
    assert initial['players']==expected['initial_players'] and final['players']==expected['final_players']
    assert final['t']==expected['ticks'] and len(actions)==expected['decisions']
    if 'replay_sha256' in expected:assert digest(directory/'game.rep')==expected['replay_sha256']
    np.savez_compressed(directory/'decisions.npz',observations=np.asarray(obs),actions=np.asarray(actions),events=np.asarray(events))
    with gzip.open(directory/'frames.json.gz','wt') as stream:json.dump(frames,stream,separators=(',',':'))
    atomic_json(directory/'original.json',expected)
    result=dict(**case,side=side,map=expected['map'],decisions=len(actions),ticks=final['t'],
                outcome='win' if expected['win'] and not expected['loss'] and not expected['timeout'] else 'loss' if expected['loss'] and not expected['win'] and not expected['timeout'] else 'draw',
                checkpoint_sha256=digest(checkpoint),replay_sha256=digest(directory/'game.rep'),
                identical_final_state_and_both_player_events=True,
                original_replay_hash_matched='replay_sha256' in expected,behavior=actual_behavior)
    atomic_json(directory/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    plan=json.loads((out/'failure-review-protocol.json').read_text());directory=out/'failure-review';directory.mkdir(exist_ok=True)
    torch.set_num_threads(1);rows=[]
    for case in plan['cases']:
        row=trace(case,directory);rows.append(row)
        atomic_json(out/'failure-review-status.json',dict(status='running',completed=len(rows),total=len(plan['cases'])))
        print(row['run'],row['case'],row['outcome'],flush=True)
    atomic_json(directory/'result.json',dict(schema=1,protocol_sha256=digest(out/'failure-review-protocol.json'),cases=rows))
    atomic_json(out/'failure-review-status.json',dict(status='complete',completed=len(rows),total=len(plan['cases'])))


if __name__=='__main__':main()
