"""Sample device-wide GPU activity and host CPU load during a declared series."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cpu_ticks():
    values=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return sum(values),values[3]+values[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',type=Path,required=True);args=parser.parse_args()
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    process=subprocess.Popen(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used,power.draw',
                              '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=subprocess.PIPE,text=True)
    previous=cpu_ticks()
    try:
        with (args.series/'resource-samples.jsonl').open('x',buffering=1) as stream:
            while not stopped:
                line=process.stdout.readline()
                if not line: raise RuntimeError(f'nvidia-smi sampler exited: {process.poll()}')
                gpu,memory,power=(float(value.strip()) for value in line.split(','))
                now=cpu_ticks();total=now[0]-previous[0];idle=now[1]-previous[1];previous=now
                state=json.loads((args.series/'status.json').read_text())
                row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),monotonic=time.monotonic(),
                         active_run=state.get('active_run'),phase=state.get('phase'),
                         gpu_busy_percent=gpu,gpu_memory_mb=memory,gpu_power_watts=power,
                         cpu_busy_percent=100*(total-idle)/total if total else 0,cpu_count=os.cpu_count())
                stream.write(json.dumps(row,allow_nan=False)+'\n')
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill();process.wait()


if __name__=='__main__': main()
