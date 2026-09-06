"""Replay original 60 Hz inputs and sample the existing 15 Hz policy contract."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from replay_data import ReplayWorld, ROOT, decode, previous

def extract(path, output, player, origin=240, episode_limit=3600):
    meta, inputs = decode(path)
    index = player-1
    if not 0 <= index < len(meta['players']): raise ValueError('Player is absent')
    target = meta['players'][index]
    if target['keyboard'] or np.any(inputs[:,index]['buttons'] & ~np.uint16(15)):
        raise ValueError('Keyboard actions cannot be represented by the gamepad policy')
    if any(1 in p['abilities'] for p in meta['players']):
        raise ValueError('Random loadouts need separate resolution before dataset admission')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    destination = output/(Path(path).stem+f'-p{player}.npz')
    if destination.exists(): raise FileExistsError(destination)
    rows, labels, ticks, alive, events = [], [], [], [], []
    history = deque(maxlen=10)
    with ReplayWorld(meta, output/(destination.stem+'.log')) as world:
        for t in range(meta['ticks']):
            if world.ended: break
            if t >= origin:
                history.append(world.pack(origin, episode_limit))
                if (t-origin)%4 == 0:
                    obs = history[0].copy()  # t-9, clamped to initial observation
                    prior = [(0,0)]*len(meta['players']) if t==origin else list(zip(inputs[t-4]['angle'], inputs[t-4]['buttons']))
                    previous(obs, prior)
                    rows.append(obs[index]); labels.append([inputs[t,index]['angle'],inputs[t,index]['buttons']])
                    ticks.append(t)
                    alive.append(world.state.players[index].alive_bodies > 0)
                    events.append([world.state.telemetry[index].arrows_fired,world.state.telemetry[index].arrow_hits])
            world.step(list(zip(inputs[t]['angle'],inputs[t]['buttons'])))
        replayed_ticks = world.state.header.episode_ticks
        final_events = [world.state.telemetry[index].arrows_fired,world.state.telemetry[index].arrow_hits]
    if not rows: raise ValueError('No post-warmup samples')
    labels = np.asarray(labels, np.int64)
    # Report what 4-tick sample-and-hold would lose. Source playback itself is
    # never quantized, and targets and own-history never use future inputs.
    changed = total = 0
    for t, action in zip(ticks,labels):
        span = inputs[t:min(t+4,replayed_ticks),index]
        changed += int(np.count_nonzero((span['angle']!=action[0]) | (span['buttons']!=action[1])))
        total += len(span)
    metadata = dict(replay=meta, player=player, origin=origin, delay=9, repeat=4,
        episode_limit=episode_limit, replayed_ticks=replayed_ticks, decisions=len(rows),
        alive_decisions=int(sum(alive)), action_resampling_mismatch=changed/max(1,total),
        final_arrows_fired=int(final_events[0]),final_arrow_hits=int(final_events[1]),
        assembly_sha256=hashlib.sha256((ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll').read_bytes()).hexdigest(),
        encoder_sha256=hashlib.sha256((ROOT/'native/rl/puffer/bopl_puffer.h').read_bytes()).hexdigest())
    np.savez_compressed(destination, observations=np.stack(rows), actions=labels,
                        ticks=np.asarray(ticks), alive=np.asarray(alive), events=np.asarray(events),
                        metadata=json.dumps(metadata))
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(file=str(destination),**{k:v for k,v in metadata.items() if k!='replay'})),flush=True)
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--player',type=int,required=True,help='1-based human player; never silently infer controller ownership')
    args=parser.parse_args()
    for path in args.replays: extract(path,args.output,args.player)

if __name__=='__main__': main()
