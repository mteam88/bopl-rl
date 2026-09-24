"""Export lightweight browser frames by replaying recorded inputs in the game."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from environment import Environment
from replay_data import decode,ReplayWorld


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();meta,inputs=decode(args.replay);out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    frames=[];origin=240
    with ReplayWorld(meta,out.with_suffix('.log')) as world:
        for t in range(meta['ticks']):
            if t>=origin and (t-origin)%4==0:
                frame=Environment.frame(world);frame['t']=max(0,frame['t']-origin);frames.append(frame)
            world.step(list(zip(inputs[t]['angle'],inputs[t]['buttons'])))
        frame=Environment.frame(world);frame['t']=max(0,frame['t']-origin)
        if not frames or frame['t']!=frames[-1]['t']:frames.append(frame)
        verified=out.with_suffix('.verified.rep');world.replay(verified)
    # Input-driven playback must preserve the native replay exactly.
    replay_hash=hashlib.sha256(verified.read_bytes()).hexdigest();verified.unlink()
    if replay_hash!=meta['sha256']:raise AssertionError('Replayed input stream differs from the saved native replay')
    value=dict(schema=1,kind='native-input-physics-replay',replay_sha256=meta['sha256'],source=meta['source'],catalog=meta['catalog'],map=meta['level'],warmup_ticks=origin,frames=frames,verified_native_replay=True)
    temporary=out.with_name(out.name+'.tmp')
    with gzip.open(temporary,'wt') as stream:json.dump(value,stream,separators=(',',':'),allow_nan=False)
    temporary.replace(out);print(json.dumps(dict(status='ready',frames=len(frames),ticks=frames[-1]['t'],replay_sha256=meta['sha256'])),flush=True)


if __name__=='__main__':main()
