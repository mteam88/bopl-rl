"""Measure C ABI step throughput (single process, includes ctypes call overhead)."""
import argparse,json,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from bopl_native import World,Action

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=100000);args=parser.parse_args()
    if args.steps<=0:raise ValueError('steps must be positive')
    actions=(Action*2)(Action(1,0),Action(128,0))
    with World() as world:
        started=time.perf_counter();resets=0
        for tick in range(args.steps):
            if tick%360==0 or world.terminal:world.reset();resets+=1
            world._check(world._lib.bopl_step(world._live(),actions,2))
        seconds=time.perf_counter()-started
    print(json.dumps(dict(world_steps=args.steps,player_steps=args.steps*2,seconds=seconds,world_steps_per_second=args.steps/seconds,
        resets=resets,host=platform.node(),machine=platform.machine(),scope='ctypes -> C ABI, two players, one thread, walk/reset workload; no policy inference or observation encoding'),indent=2))
if __name__=='__main__':main()
