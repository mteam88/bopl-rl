"""Correctness checks for pipelined environment stepping and CUDA-graph inference.

Phase A (transport): the same fixed action sequence is stepped through the
vector with 1, 2 and 4 pipeline groups, with groups waited in launch order and
in reverse, with the caller's action tensor overwritten right after launch,
and with BOPL_GROUP_JITTER_US perturbing thread interleavings. Observations,
rewards, terminals, telemetry and host terminals must be bit-identical to the
single-group synchronous path at every decision.

Phase B (rollout): train.py is run with --pipeline-serial (each group waited
right after launch: identical (t,group) inference and RNG order, no overlap)
and without it; the complete first-update rollout buffers and all metrics must
be bit-identical, also under jitter. Phase C checks alignment invariants on
each dump: obs[t+1] carries the own-action history of actions[t] for every
agent that did not reset, resets[t+1]==dones[t], recurrent state is zero after
a reset, and segment memories equal the state at segment starts.

Phase D: --cuda-graphs 1 must reproduce --cuda-graphs 0 bit-exactly (rollout
buffers and metrics), for 1 and 2 groups, across a checkpoint that adds a
second historical policy. Phase E: groups=1 metrics equal a reference run of
the original loop (which is run-to-run deterministic). Phase F: the three
historical-opponent sampling modes (BOPL_OPPONENT_SAMPLING=graph, slim,
distribution) give bit-identical rollouts and metrics with 2 groups and graphs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent

def transport(out,workers,decisions,limit):
    from environment import configure
    configure(out/'workers')
    import torch
    from vector import Vector
    from telemetry import FIELDS
    rng=np.random.default_rng(1)
    plan=[np.stack([rng.integers(256,size=2*workers),rng.integers(16,size=2*workers)],axis=-1).astype(np.float32) for _ in range(decisions)]
    def run(groups,mode):
        vec=Vector(workers,73,limit,groups=groups)
        assert vec.step_mode()=='sampled', 'game build rejected the sampled step'
        obs=[];rew=[];don=[];tel=[];cpu=[]
        for t in range(decisions):
            actions=torch.as_tensor(plan[t],device='cuda')
            if groups==1: vec.step(actions)
            else:
                order=list(range(groups))
                for g in order: vec.step_group(g,actions[vec.groups[g]])
                if mode=='overwrite': actions.fill_(7)   # the caller's tensor is not part of the hand-off
                if mode=='reverse': order=order[::-1]
                for g in order:
                    assert vec.wait_group(g), 'the first wait after a launch must collect it'
                    assert not vec.wait_group(g), 'a second wait must find nothing to collect'
                    assert not vec.pending(g)
            obs.append(vec.obs.clone());rew.append(vec.rewards.clone());don.append(vec.dones.clone())
            tel.append(vec.telemetry.copy());cpu.append(vec.cpu_dones.clone())
        log=vec.log()
        # Checked after real stepping: capability discovery happens on the first sampled request.
        assert vec.step_mode()=='sampled', f'transport is {vec.step_mode()}, not the native sampled step'
        profile=vec.group_profile()
        if groups>1:
            # Close with a step in flight: must wait, not free buffers under the thread.
            vec.step_group(0,torch.as_tensor(plan[0],device='cuda')[vec.groups[0]])
        vec.close()
        return dict(obs=torch.stack(obs),rewards=torch.stack(rew),dones=torch.stack(don),telemetry=np.stack(tel),cpu_dones=torch.stack(cpu)),log,profile
    reference,ref_log,_=run(1,'sync')
    # vec.log()['n'] counts finished player episodes (two per match).
    results=dict(reference_log=ref_log,matches=float(ref_log.get('n',0))/2,cases=[])
    for groups,mode,jitter in [(2,'inorder',0),(4,'inorder',0),(2,'reverse',0),(4,'reverse',0),(2,'overwrite',0),(2,'inorder',400),(4,'reverse',900),(2,'overwrite',1500)]:
        os.environ['BOPL_GROUP_JITTER_US']=str(jitter)
        got,log,profile=run(groups,mode)
        os.environ.pop('BOPL_GROUP_JITTER_US')
        for k in ('obs','rewards','dones','cpu_dones'):
            if not torch.equal(got[k],reference[k]): raise AssertionError(f'{k} differs: groups={groups} mode={mode} jitter={jitter}')
        if not np.array_equal(got['telemetry'],reference['telemetry']): raise AssertionError(f'telemetry differs: groups={groups} mode={mode}')
        for k in ref_log:
            if k in ('sps','environment_ticks'): continue
            if k=='step_mode':
                if log[k]!='sampled' or ref_log[k]!='sampled': raise AssertionError(f'step mode {log[k]}/{ref_log[k]}: groups={groups} mode={mode}')
            elif abs(float(log.get(k,0))-float(ref_log[k]))>1e-6: raise AssertionError(f'log {k} differs: groups={groups} mode={mode}')
        results['cases'].append(dict(groups=groups,mode=mode,jitter_us=jitter,status='exact',launches=[p['launches'] for p in profile.values()]))
    assert results['matches']>=2*workers, 'the transport check must cover autoresets'
    return results

def train(out,name,workers,steps,extra,env=None):
    run=out/name; dump=out/f'{name}-rollout.pt'
    environment=dict(os.environ,BOPL_DUMP_ROLLOUT=str(dump),**(env or {}))
    command=[sys.executable,str(HERE/'train.py'),'--output',str(run),'--workers',str(workers),'--steps',str(steps),'--seed','73',
             '--no-eval','--no-video','--episode-ticks','200','--checkpoint-every',str(steps//3),*extra]
    started=time.monotonic()
    with (out/f'{name}.log').open('w') as log:
        log.write(' '.join(command)+'\n'); log.flush()
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=environment,check=True)
    import torch
    records=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    return dict(dump=torch.load(dump,map_location='cuda'),metrics=records,seconds=time.monotonic()-started,
                sps=float(np.mean([r['sps'] for r in records])),config=json.loads((run/'config.json').read_text()))

STRIP={'elapsed','sps','rollout_seconds','train_seconds','snapshot_seconds','steppers','cuda_memory_mb','cuda_reserved_mb','pipeline','env'}
ENV_STRIP={'sps','step_mode','sampled_native'}   # timing and transport metadata, absent from the original loop
def comparable(records):
    return [{k:v for k,v in r.items() if k not in STRIP}|{'env':{k:v for k,v in r['env'].items() if k not in ENV_STRIP}} for r in records]

def same_dump(a,b,label):
    import torch
    for k in a:
        if not torch.equal(a[k],b[k]): raise AssertionError(f'{label}: rollout {k} differs')

def alignment(dump,workers,length):
    import torch
    obs=dump['obs'];actions=dump['actions'];dones=dump['dones'].bool();resets=dump['resets'];states=dump['states'];memories=dump['memories']
    tmax=obs.shape[0]
    if not torch.equal(resets[1:],dones[:-1]): raise AssertionError('resets[t+1] != dones[t]')
    if not torch.equal(dump['final_reset'],dones[-1]): raise AssertionError('final reset flag != dones[-1]')
    # Own-action history: obs[t+1][-7:] encodes actions[t] unless the agent reset.
    angle=actions[:-1,:,0].double();buttons=actions[:-1,:,1].long()
    radians=(angle-1)*6.283185307179586/255.0   # same double arithmetic as c_step
    zero=torch.zeros_like(angle,dtype=torch.float32)
    expected=torch.stack([(angle==0).float(),torch.where(angle>0,torch.cos(radians).float(),zero),
                          torch.where(angle>0,torch.sin(radians).float(),zero),
                          *[((buttons>>bit)&1).float() for bit in range(4)]],dim=-1)
    got=obs[1:,:,-7:]
    alive=~dones[:-1]
    if not torch.equal(got[alive],expected[alive]): raise AssertionError('obs[t+1] own-action history does not match actions[t]')
    reset_rows=got[dones[:-1]]
    if len(reset_rows) and not (torch.equal(reset_rows[:,0],torch.ones_like(reset_rows[:,0])) and torch.equal(reset_rows[:,1:],torch.zeros_like(reset_rows[:,1:]))):
        raise AssertionError('reset agents did not receive the fresh-episode history')
    # Recurrent state is zeroed for reset agents before their next inference.
    reset_states=states[1:][dones[:-1].view(tmax-1,1,-1,1).expand_as(states[1:])]
    if not torch.equal(reset_states,torch.zeros_like(reset_states)): raise AssertionError('recurrent state not reset after done')
    for k in range(tmax//length):
        if not torch.equal(memories[k],states[k*length]): raise AssertionError('segment memory != state at segment start')
    ticks=dump['events'][:,:,-1]
    if not ((ticks<=4)&(ticks>=1)).all(): raise AssertionError('each decision must advance 1..4 ticks')
    masks=dump['masks']
    learner=masks.view(tmax,workers,2)
    if not learner[:,0::2].all(): raise AssertionError('even workers must be fully learner-controlled')
    if not (learner[:,1::2].sum(-1)==1).all(): raise AssertionError('odd workers must have exactly one learner agent')
    return dict(decisions=int(tmax*obs.shape[1]),resets=int(dones.sum()),reset_agents_checked=int(dones[:-1].sum()))

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--reference-metrics',type=Path,help='metrics.jsonl of an original-loop run to compare a groups=1 run against')
    parser.add_argument('--reference-args',default='--workers 16 --steps 98304 --pipeline-groups 1 --cuda-graphs 0',help='train.py arguments reproducing the reference run')
    parser.add_argument('--skip-transport',action='store_true')
    opt=parser.parse_args()
    out=opt.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,workers=opt.workers,phases={})
    def save(): (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    if not opt.skip_transport:
        report['phases']['A_transport']=transport(out/'transport',opt.workers,360,400); save()
        print('A transport: exact',flush=True)
    w=opt.workers; n=2*w; steps=3*128*n
    runs={}
    runs['g2-serial']=train(out,'g2-serial',w,steps,['--pipeline-groups','2','--pipeline-serial','--cuda-graphs','0'])
    runs['g2']=train(out,'g2',w,steps,['--pipeline-groups','2','--cuda-graphs','0'])
    runs['g2-jitter']=train(out,'g2-jitter',w,steps,['--pipeline-groups','2','--cuda-graphs','0'],env={'BOPL_GROUP_JITTER_US':'700'})
    runs['g4-serial']=train(out,'g4-serial',w,steps,['--pipeline-groups','4','--pipeline-serial','--cuda-graphs','0'])
    runs['g4']=train(out,'g4',w,steps,['--pipeline-groups','4','--cuda-graphs','0'])
    runs['g4-jitter']=train(out,'g4-jitter',w,steps,['--pipeline-groups','4','--cuda-graphs','0'],env={'BOPL_GROUP_JITTER_US':'1200'})
    for a,b in (('g2-serial','g2'),('g2-serial','g2-jitter'),('g4-serial','g4'),('g4-serial','g4-jitter')):
        same_dump(runs[a]['dump'],runs[b]['dump'],f'{a} vs {b}')
        if comparable(runs[a]['metrics'])!=comparable(runs[b]['metrics']): raise AssertionError(f'{a} vs {b}: metrics differ')
    report['phases']['B_serial_vs_async']=dict(status='exact',pairs=['g2-serial=g2','g2-serial=g2-jitter','g4-serial=g4','g4-serial=g4-jitter'],
        updates=len(runs['g2']['metrics']),matches={k:v['metrics'][-1]['matches'] for k,v in runs.items()}); save()
    print('B serial vs async: exact',flush=True)
    runs['g1']=train(out,'g1',w,steps,['--pipeline-groups','1','--cuda-graphs','0'])
    report['phases']['C_alignment']={name:alignment(r['dump'],w,32) for name,r in runs.items()}; save()
    print('C alignment: pass',flush=True)
    runs['g1-graphs']=train(out,'g1-graphs',w,steps,['--pipeline-groups','1','--cuda-graphs','1'])
    runs['g2-graphs']=train(out,'g2-graphs',w,steps,['--pipeline-groups','2','--cuda-graphs','1'])
    graph_status={}
    for a,b in (('g1','g1-graphs'),('g2','g2-graphs')):
        same_dump(runs[a]['dump'],runs[b]['dump'],f'{a} vs {b}')
        graph_status[f'{a}={b}']='exact rollout; metrics '+('exact' if comparable(runs[a]['metrics'])==comparable(runs[b]['metrics']) else 'DIFFER')
    if any('DIFFER' in v for v in graph_status.values()): raise AssertionError(f'CUDA graph metrics differ: {graph_status}')
    report['phases']['D_cuda_graphs']=dict(status='exact',pairs=graph_status,pool_versions=runs['g1-graphs']['metrics'][-1]['pool_steps']); save()
    print('D cuda graphs: exact',flush=True)
    if opt.reference_metrics:
        reference=[json.loads(l) for l in opt.reference_metrics.read_text().splitlines()]
        run=out/'g1-reference'
        with (out/'g1-reference.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'train.py'),'--output',str(run),'--seed','73','--no-eval','--no-video',*opt.reference_args.split()],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        mine=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        if comparable(mine)!=comparable(reference): raise AssertionError('groups=1 metrics differ from the original-loop reference')
        report['phases']['E_original_loop']=dict(status='exact',updates=len(mine),reference=str(opt.reference_metrics)); save()
        print('E original loop: exact',flush=True)
    for mode in ('slim','distribution'):
        runs[f'g2-graphs-{mode}']=train(out,f'g2-graphs-{mode}',w,steps,['--pipeline-groups','2','--cuda-graphs','1'],env={'BOPL_OPPONENT_SAMPLING':mode})
        same_dump(runs['g2-graphs']['dump'],runs[f'g2-graphs-{mode}']['dump'],f'g2-graphs vs {mode}')
        if comparable(runs['g2-graphs']['metrics'])!=comparable(runs[f'g2-graphs-{mode}']['metrics']): raise AssertionError(f'opponent sampling {mode}: metrics differ')
    report['phases']['F_opponent_sampling']=dict(status='exact',modes=['graph','slim','distribution']); save()
    print('F opponent sampling modes: exact',flush=True)
    report['status']='pass'; save(); print(json.dumps(report['phases'],default=str)[:2000])

if __name__=='__main__': main()
