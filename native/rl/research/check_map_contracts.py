"""Run native map-selection/logging/overflow checks without a game or GPU."""
import argparse
import json
from pathlib import Path
import resource
import signal
import subprocess

ROOT=Path(__file__).resolve().parents[3]
RL=ROOT/'native/rl'
PUFFER=ROOT/'native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    commands=[['cc','-O2','-std=c11','-I'+str(RL),str(RL/'research/check_map_sampling.c'),'-lm','-o',str(out/'sampling')],
              ['cc','-O2','-DPRECISION_FLOAT','-fopenmp','-ffunction-sections','-fdata-sections','-Wl,--gc-sections',
               '-I'+str(PUFFER/'src'),'-I'+str(RL),str(RL/'research/check_map_logs.c'),str(RL/'bopl_rl.c'),
               '-lm','-lpthread','-o',str(out/'logs')]]
    for command in commands: subprocess.run(command,check=True)
    for name in ('sampling','logs'): subprocess.run([str(out/name)],check=True)
    def no_core(): resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    overflow=subprocess.run([str(out/'sampling'),'overflow'],capture_output=True,text=True,preexec_fn=no_core)
    assert overflow.returncode==-signal.SIGABRT and 'observation overflow' in overflow.stderr
    report=dict(status='pass',selection_and_default_rng=True,all_map_log_keys=True,overflow_before_episode_end=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
