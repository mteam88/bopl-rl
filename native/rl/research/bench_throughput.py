"""Matched throughput benchmark for research/train.py with GPU and CPU sampling.

Runs the actual training loop (`--no-eval --no-video`) for a fixed decision
budget at each worker count, interleaving worker counts across repeats so that
slow drift of the machine affects every arm equally. While each run executes,
a sampler polls nvidia-smi (utilization, power, memory, SM clock) and /proc/stat
(busy cores) four times per second. Samples are attributed to the steady loop
window, which starts when the first metrics record appears and ends when the
trainer exits.

Throughput is reported as decisions per second of collection-plus-PPO loop
time (the same quantity train.py logs as `sps`), plus wall time per run.
Use a new --output directory per benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent

def proc_stat():
    with open('/proc/stat') as f: fields=f.readline().split()
    total=sum(int(x) for x in fields[1:]); idle=int(fields[4])+int(fields[5])
    return total,idle

class Sampler(threading.Thread):
    def __init__(self,metrics,interval=0.25):
        super().__init__(daemon=True); self.metrics=metrics; self.interval=interval
        self.samples=[]; self.stop=threading.Event(); self.loop_started=None
    def run(self):
        previous=proc_stat(); hz=os.sysconf('SC_CLK_TCK')
        while not self.stop.is_set():
            time.sleep(self.interval)
            now=time.monotonic()
            if self.loop_started is None and self.metrics.exists() and self.metrics.stat().st_size>0:
                self.loop_started=now
            try:
                text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm,temperature.gpu',
                                              '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                util,power,memory,clock,temperature=[float(x) for x in text.split(',')]
            except Exception: continue
            current=proc_stat()
            busy=((current[0]-current[1])-(previous[0]-previous[1]))/max(1,current[0]-previous[0])*os.cpu_count()
            previous=current
            self.samples.append(dict(t=now,gpu_util=util,gpu_power_w=power,gpu_memory_mb=memory,sm_clock_mhz=clock,
                                     gpu_temperature_c=temperature,busy_cores=busy,load1=os.getloadavg()[0]))

def summarize(values):
    if not values: return None
    values=sorted(values)
    return dict(mean=statistics.fmean(values),median=statistics.median(values),
                p10=values[int(.1*len(values))],p90=values[min(len(values)-1,int(.9*len(values)))],n=len(values))

def run_once(python,out,workers,decisions,seed,extra,env):
    metrics=out/'metrics.jsonl'
    command=[python,str(HERE/'train.py'),'--output',str(out),'--workers',str(workers),'--steps',str(decisions),
             '--seed',str(seed),'--no-eval','--no-video',*extra]
    log=out.with_suffix('.log').open('w')
    log.write(' '.join(shlex.quote(x) for x in command)+'\n'); log.flush()
    sampler=Sampler(metrics); sampler.start()
    started=time.monotonic()
    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    finished=time.monotonic(); sampler.stop.set(); sampler.join(); log.close()
    if process.returncode: raise RuntimeError(f'{out.name} failed with {process.returncode}; see {log.name}')
    records=[json.loads(line) for line in metrics.read_text().splitlines()]
    # Count only samples actually consumed by PPO, including on resumed runs.
    learner_decisions=sum(r['behavior']['decisions'] for r in records)
    loop=sum(r['rollout_seconds']+r['train_seconds'] for r in records)
    rollout=sum(r['rollout_seconds'] for r in records)
    snapshot=sum(r.get('snapshot_seconds',0) for r in records)
    steps=records[-1]['step']
    result=json.loads((out/'result.json').read_text())
    window=[s for s in sampler.samples if sampler.loop_started is not None and s['t']>=sampler.loop_started]
    # active_seconds: train.py's own clock from pool/graph setup through the
    # final update (result.elapsed), including snapshot and graph-capture cost.
    return dict(name=out.name,workers=workers,seed=seed,decisions=steps,updates=len(records),matches=result['matches'],
                learner_decisions=learner_decisions,
                learner_dps_active=learner_decisions/result['elapsed'],
                learner_dps_wall=learner_decisions/(finished-started),
                loop_seconds=loop,rollout_seconds=rollout,train_seconds=loop-rollout,snapshot_seconds=snapshot,
                active_seconds=result['elapsed'],pool_versions=records[-1].get('pool_versions'),steppers=records[-1].get('steppers'),
                decisions_per_second=steps/loop,decisions_per_second_active=steps/result['elapsed'],
                decisions_per_second_wall=steps/(finished-started),wall_seconds=finished-started,
                startup_seconds=json.loads((out/'config.json').read_text())['startup_seconds'],
                step_mode=json.loads((out/'config.json').read_text()).get('step_mode'),
                cuda_memory_mb=records[-1]['cuda_memory_mb'],
                window_seconds=(window[-1]['t']-window[0]['t']) if len(window)>1 else 0,
                gpu_util=summarize([s['gpu_util'] for s in window]),gpu_power_w=summarize([s['gpu_power_w'] for s in window]),
                gpu_memory_mb=summarize([s['gpu_memory_mb'] for s in window]),sm_clock_mhz=summarize([s['sm_clock_mhz'] for s in window]),
                busy_cores=summarize([s['busy_cores'] for s in window]),load1=summarize([s['load1'] for s in window]),
                samples=sampler.samples)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',default='16,24,32')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--decisions',type=int,default=98304)
    parser.add_argument('--seed',type=int,default=73)
    parser.add_argument('--label',default='')
    parser.add_argument('--arms',default='',help='name=extra train.py args; comma separated, e.g. "sync=--pipeline-groups 1 --cuda-graphs 0,pipe=--pipeline-groups 2". Pass flags explicitly; an empty arm uses train.py defaults')
    parser.add_argument('extra',nargs='*',help='extra train.py arguments for every run (after --)')
    opt=parser.parse_args()
    out=opt.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    workers=[int(x) for x in opt.workers.split(',')]
    arms=[('base',[])] if not opt.arms else [(a.split('=',1)[0],shlex.split(a.split('=',1)[1])) for a in opt.arms.split(',')]
    python=sys.executable; env=dict(os.environ)
    runs=[]; summary=out/'summary.json'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    def write():
        table={}
        for arm,_ in arms:
            for w in workers:
                rows=[r for r in runs if r['workers']==w and r['arm']==arm]
                if not rows: continue
                table[f'{arm}-w{w}']=dict(arm=arm,workers=w,runs=[r['name'] for r in rows],
                    median_learner_dps_active=statistics.median(r['learner_dps_active'] for r in rows),
                    median_learner_dps_wall=statistics.median(r['learner_dps_wall'] for r in rows),
                    decisions_per_second=[r['decisions_per_second'] for r in rows],
                    median_decisions_per_second=statistics.median(r['decisions_per_second'] for r in rows),
                    median_decisions_per_second_active=statistics.median(r['decisions_per_second_active'] for r in rows),
                    median_decisions_per_second_wall=statistics.median(r['decisions_per_second_wall'] for r in rows),
                    median_snapshot_seconds=statistics.median(r['snapshot_seconds'] for r in rows),
                    median_gpu_util=statistics.median(r['gpu_util']['mean'] for r in rows if r['gpu_util']),
                    median_gpu_power_w=statistics.median(r['gpu_power_w']['mean'] for r in rows if r['gpu_power_w']),
                    median_busy_cores=statistics.median(r['busy_cores']['mean'] for r in rows if r['busy_cores']),
                    median_wall_seconds=statistics.median(r['wall_seconds'] for r in rows),
                    collection_share=statistics.median(r['rollout_seconds']/r['loop_seconds'] for r in rows))
        summary.write_text(json.dumps(dict(schema=1,label=opt.label,gpu=gpu,cpu_count=os.cpu_count(),decisions=opt.decisions,
            seed=opt.seed,repeats=opt.repeats,extra=opt.extra,arms={a:e for a,e in arms},table=table,
            runs=[{k:v for k,v in r.items() if k!='samples'} for r in runs]),indent=2)+'\n')
    for repeat in range(opt.repeats):
        for w in workers:
            for arm,extra in arms:
                name=f'{arm}-w{w}-{repeat+1}'
                print(f'== {name}',flush=True)
                result=run_once(python,out/name,w,opt.decisions,opt.seed,[*extra,*opt.extra],env); result['arm']=arm
                (out/f'{name}-samples.json').write_text(json.dumps(result['samples'])+'\n')
                runs.append(result); write()
                print(json.dumps({k:v for k,v in result.items() if k in ('name','decisions_per_second','decisions_per_second_active','decisions_per_second_wall','snapshot_seconds','pool_versions','wall_seconds','gpu_util','gpu_power_w','busy_cores')}),flush=True)
    print(summary.read_text())

if __name__=='__main__': main()
