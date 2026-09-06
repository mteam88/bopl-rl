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

def capture(root,out,games):
    sys.path.insert(0,str(root/'native/rl/research'))
    # This also supports the previous checkout, whose module lacks telemetry.
    import environment as E
    E.configure(out/'workers')
    result=[]
    with E.Environment(seed=73,limit=3600) as env:
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
                actions=np.stack([rng.integers(256,size=2),rng.integers(16,size=2)],axis=-1)
                env.step(actions); record();decisions+=1
                if hasattr(env,'telemetry'): events.append(env.telemetry.copy())
            replay=out/f'episode-{game}.rep';env.replay(replay)
            row=dict(seed=seed,decisions=decisions,ticks=env.state.header.episode_ticks,
                     state_sha256=state_hash.hexdigest(),observations_sha256=obs_hash.hexdigest(),
                     replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
            if hasattr(env,'step_mode'):
                # Checked after real stepping: an old game build would silently
                # fall back to two roundtrips and must not pass as the new IPC path.
                row['step_mode']=env.step_mode
                if env.step_mode=='fallback': raise AssertionError('Game build rejected the sampled step command')
            if events:
                from telemetry import EVENT_FIELDS,FIELDS
                total=np.asarray(events).sum(axis=0,dtype=np.float64)
                actual=np.array([[getattr(env.state.telemetry[p],k) for k in EVENT_FIELDS] for p in range(2)])
                np.testing.assert_array_equal(total[:,:len(EVENT_FIELDS)],actual)
                deaths=np.array([p.deaths for p in env.state.players[:2]])-initial_deaths
                np.testing.assert_array_equal(actual[:,5:9].sum(axis=1),deaths)
                np.testing.assert_array_equal(total[:,FIELDS.index('ticks')],env.state.header.episode_ticks)
                np.testing.assert_array_equal(total[:,FIELDS.index('episodes')],1)
                np.testing.assert_allclose(total[:,FIELDS.index('reward_deaths')],-0.1*deaths,rtol=1e-6)
                row['totals']=dict(zip(FIELDS,total.sum(axis=0).tolist()))
            result.append(row)
        if hasattr(env,'telemetry'):
            env.reset(18)
            np.testing.assert_array_equal(env.telemetry,0)
            assert all(getattr(env.state.telemetry[p],k)==0 for p in range(2) for k in EVENT_FIELDS)
    (out/'capture.json').write_text(json.dumps(result,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--games',type=int,default=12)
    parser.add_argument('--capture-root',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.capture_root:
        capture(args.capture_root.resolve(),args.output,args.games);return
    subprocess.run([sys.executable,str(__file__),'--output',str(args.output/'candidate'),
                    '--capture-root',str(ROOT),'--games',str(args.games)],check=True)
    candidate=json.loads((args.output/'candidate/capture.json').read_text())
    totals={k:sum(row['totals'][k] for row in candidate) for k in candidate[0]['totals']}
    for k in ('dash_activations','bow_activations','platform_activations','arrows_fired','arrow_hits','deaths_environment','deaths_opponent'):
        if totals[k]<=0: raise AssertionError(f'Missing live coverage for {k}; extend the deterministic game corpus')
    assembly=Path('Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll')
    report=dict(status='pass',games=args.games,totals=totals,event_conservation='pass',reset='pass',
                step_mode=sorted(set(row.get('step_mode','unknown') for row in candidate)),
                candidate_assembly_sha256=hashlib.sha256((ROOT/assembly).read_bytes()).hexdigest(),episodes=candidate)
    if args.baseline:
        subprocess.run([sys.executable,str(__file__),'--output',str(args.output/'baseline'),
                        '--capture-root',str(args.baseline.resolve()),'--games',str(args.games)],check=True)
        baseline=json.loads((args.output/'baseline/capture.json').read_text())
        for old,new in zip(baseline,candidate,strict=True):
            for k in ('seed','decisions','ticks','state_sha256','observations_sha256','replay_sha256'):
                if old[k]!=new[k]: raise AssertionError(f'Baseline divergence for seed {old["seed"]}: {k}')
        report['baseline_gameplay_observations_replays']='exact match'
        report['baseline_assembly_sha256']=hashlib.sha256((args.baseline/assembly).read_bytes()).hexdigest()
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
