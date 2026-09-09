"""Check live event conservation and exact gameplay/observation parity with a baseline build.

The baseline comparison hashes fixed-point state, every delayed policy observation
and the lossless input replay. The protocol version and new telemetry are excluded
from the state hash; existing gameplay fields are not rounded or omitted.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def capture(out,games,game=None):
    import environment as E
    from layout import ACT_AIM,ACT_BUTTONS
    from telemetry import EVENT_FIELDS,FIELDS
    E.configure(out/'workers',game=game)
    result=[]
    sparse=dict(win=1,loss=-1,kill=0,death=0,timeout=-0.5)
    with E.Environment(seed=73,limit=3600,reward=sparse) as env:
        for game in range(games):
            seed=10001+game; env.reset(seed)
            rng=np.random.default_rng(9182+seed)
            obs_hash=hashlib.sha256(); state_hash=hashlib.sha256()
            initial_deaths=np.array([p.deaths for p in env.state.players[:2]])
            events=[]; decisions=0
            def record():
                s=env.state
                state_hash.update(C.string_at(C.addressof(s.header),C.sizeof(E.Header))[4:])
                state_hash.update(C.string_at(C.addressof(s.players),s.header.player_count*C.sizeof(E.Player)))
                state_hash.update(C.string_at(s.entities,s.header.entity_count*C.sizeof(E.Entity)))
                state_hash.update(C.string_at(s.slimes,s.header.slime_count*C.sizeof(E.Slime)))
                obs_hash.update(env.obs.tobytes())
            record()
            while not env.ended:
                # Random aim and buttons; hold index 0 (length 1) so no decision is latched.
                actions=np.stack([rng.integers(ACT_AIM,size=2),rng.integers(ACT_BUTTONS,size=2),np.zeros(2,np.int64)],axis=-1)
                env.step(actions); record();decisions+=1
                events.append(env.telemetry.copy())
            replay=out/f'episode-{game}.rep';env.replay(replay)
            row=dict(seed=seed,decisions=decisions,ticks=env.state.header.episode_ticks,
                     state_sha256=state_hash.hexdigest(),observations_sha256=obs_hash.hexdigest(),
                     replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
            if events:
                total=np.asarray(events).sum(axis=0,dtype=np.float64)
                actual=np.array([[getattr(env.state.telemetry[p],k) for k in EVENT_FIELDS] for p in range(2)])
                np.testing.assert_array_equal(total[:,:len(EVENT_FIELDS)],actual)
                deaths=np.array([p.deaths for p in env.state.players[:2]])-initial_deaths
                np.testing.assert_array_equal(actual[:,5:9].sum(axis=1),deaths)
                np.testing.assert_array_equal(total[:,FIELDS.index('ticks')],env.state.header.episode_ticks)
                np.testing.assert_array_equal(total[:,FIELDS.index('episodes')],1)
                np.testing.assert_array_equal(total[:,FIELDS.index('reward_deaths')],0)
                np.testing.assert_array_equal(total[:,FIELDS.index('reward_kills')],0)
                truncated=bool(env.state.header.flags&2)
                expected=[-0.5 if truncated else (1 if p.flags&2 else -1) for p in env.state.players[:2]]
                np.testing.assert_array_equal(total[:,FIELDS.index('reward_outcome')],expected)
                np.testing.assert_array_equal(total[:,FIELDS.index('timeouts')],int(truncated))
                row['totals']=dict(zip(FIELDS,total.sum(axis=0).tolist()))
            result.append(row)
        env.reset(18)
        np.testing.assert_array_equal(env.telemetry,0)
        assert all(getattr(env.state.telemetry[p],k)==0 for p in range(2) for k in EVENT_FIELDS)
    # Force a short timeout so the sparse timeout contract is always exercised.
    with E.Environment(seed=18,limit=4,reward=sparse) as env:
        env.step([[0,0,0],[0,0,0]])
        assert env.ended and env.state.header.flags&2
        np.testing.assert_array_equal(env.telemetry[:,FIELDS.index('reward_outcome')],-0.5)
        np.testing.assert_array_equal(env.telemetry[:,FIELDS.index('reward_kills')],0)
        np.testing.assert_array_equal(env.telemetry[:,FIELDS.index('reward_deaths')],0)
        np.testing.assert_array_equal(env.telemetry[:,FIELDS.index('timeouts')],1)
    (out/'capture.json').write_text(json.dumps(dict(episodes=result,game_identity=E.game_identity()),indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--game',type=Path)
    parser.add_argument('--baseline-game',type=Path,help='Compare another compiled player using the same native environment')
    parser.add_argument('--games',type=int,default=12)
    parser.add_argument('--capture',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.games<1: parser.error('--games must be positive')
    if args.capture:
        capture(args.output,args.games,args.game);return
    def run(label,game):
        command=[sys.executable,str(__file__),'--output',str(args.output/label),'--capture','--games',str(args.games)]
        if game: command+=['--game',str(game.resolve())]
        subprocess.run(command,check=True)
        return json.loads((args.output/label/'capture.json').read_text())
    candidate_capture=run('candidate',args.game)
    candidate=candidate_capture['episodes']
    totals={k:sum(row['totals'][k] for row in candidate) for k in candidate[0]['totals']}
    for k in ('dash_activations','bow_activations','platform_activations','arrows_fired','arrow_hits','deaths_environment','deaths_opponent'):
        if totals[k]<=0: raise AssertionError(f'Missing live coverage for {k}; extend the deterministic game corpus')
    report=dict(status='pass',games=args.games,totals=totals,event_conservation='pass',reset='pass',
                sparse_reward='pass',forced_timeout='pass',
                candidate_game_identity=candidate_capture['game_identity'],episodes=candidate)
    if args.baseline_game:
        baseline_capture=run('baseline',args.baseline_game)
        baseline=baseline_capture['episodes']
        for old,new in zip(baseline,candidate,strict=True):
            for k in ('seed','decisions','ticks','state_sha256','observations_sha256','replay_sha256'):
                if old[k]!=new[k]: raise AssertionError(f'Baseline divergence for seed {old["seed"]}: {k}')
        report['baseline_gameplay_observations_replays']='exact match'
        report['baseline_game_identity']=baseline_capture['game_identity']
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
