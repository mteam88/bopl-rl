"""Execute the frozen nine-run comparison and all final-checkpoint evaluations."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from train import atomic_json, ROOT
from prepare_league import digest


def verify(out, protocol):
    if digest(out/'protocol.json')!=(out/'protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol changed')
    for name,expected in protocol['training_source_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise ValueError(f'Frozen source changed: {name}')
    for name,key in [('training-bots.json','training_bots_sha256'),('bow-initial.pt','initialization_sha256')]:
        if digest(out/name)!=protocol[key]:raise ValueError(f'Frozen input changed: {name}')
    if digest(ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll')!=protocol['assembly_sha256']:
        raise ValueError('Game assembly changed')
    if digest(ROOT/'native/artifacts/rl/libbopl_research.so')!=protocol['research_library_sha256']:
        raise ValueError('Observation library changed')


def schedule(protocol, suite, checkpoint):
    source=dict(kind='policy',path=str(checkpoint.resolve()),sha256=digest(checkpoint))
    spec=protocol[suite];tasks=[]
    for m in spec['maps']:
        for i,bot in enumerate(spec['bots']):
            for pair in range(spec['pairs_per_cell']):
                for side in (0,1):
                    name=f"{bot['name']}-m{m}-p{pair}-s{side}"
                    tasks.append(dict(name=name,group=f"{bot['name']}-m{m}",map=m,side=side,
                                      seed=spec['seed_base']+m*1000+i*100+pair,
                                      learner=source,opponent=dict(kind='script',spec=bot),save=pair==0))
    return dict(schema=1,limit=protocol['episode_ticks'],tasks=tasks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('train','evaluate','all'),default='all')
    parser.add_argument('--evaluation-workers',type=int,default=6)
    args=parser.parse_args();out=args.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text());verify(out,protocol)
    started=time.monotonic()
    for run in protocol['runs']:
        path=out/'runs'/run['name']
        if args.phase in ('train','all'):
            verify(out,protocol)
            if not (path/'result.json').exists():
                if path.exists():raise ValueError(f'Incomplete run needs inspection: {path}')
                path.parent.mkdir(exist_ok=True)
                command=[sys.executable,str(Path(__file__).with_name('train.py')),
                         '--output',str(path),'--seed',str(run['seed']),
                         '--learner-steps',str(protocol['learner_decisions']),
                         '--workers',str(protocol['workers']),'--horizon',str(protocol['horizon']),
                         '--sequence',str(protocol['sequence']),'--epochs',str(protocol['epochs']),
                         '--lr',str(protocol['learning_rate']),'--episode-ticks',str(protocol['episode_ticks']),
                         '--training-maps',','.join(map(str,protocol['training_maps'])),
                         '--no-eval','--no-video']
                if run['condition']!='A':command+=['--scripted-opponents',str(out/'training-bots.json'),
                                                  '--scripted-fraction',str(protocol['scripted_fraction_of_fixed_opponent_assignments'])]
                if run['condition']=='C':command+=['--initialize',str(out/'bow-initial.pt')]
                atomic_json(out/'series-status.json',dict(phase='train',run=run['name'],elapsed=time.monotonic()-started))
                print('TRAIN',run['name'],flush=True)
                with (out/(run['name']+'.log')).open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
            result=json.loads((path/'result.json').read_text())
            if result['status']!='complete' or result['learner_steps']<protocol['learner_decisions']:
                raise ValueError('Run did not satisfy its learner budget')
    if args.phase in ('evaluate','all'):
        for run in protocol['runs']:
            path=out/'runs'/run['name']
            result=json.loads((path/'result.json').read_text())
            if result['status']!='complete':raise ValueError('Training incomplete')
            checkpoint=path/'checkpoints'/Path(result['checkpoint']).name
            for suite in ('development','final'):
                verify(out,protocol)
                destination=out/'evaluations'/run['name']/suite
                if (destination/'result.json').exists():continue
                plan=schedule(protocol,suite,checkpoint)
                schedule_path=out/(run['name']+'-'+suite+'-schedule.json')
                atomic_json(schedule_path,plan)
                atomic_json(out/'series-status.json',dict(phase='evaluate',run=run['name'],suite=suite,
                                                         elapsed=time.monotonic()-started))
                print('EVALUATE',run['name'],suite,flush=True)
                with (out/(run['name']+'-'+suite+'.log')).open('w') as log:
                    subprocess.run([sys.executable,str(Path(__file__).with_name('league_evaluate.py')),
                                    '--schedule',str(schedule_path),'--output',str(destination),
                                    '--workers',str(args.evaluation_workers)],stdout=log,stderr=subprocess.STDOUT,
                                    check=True,cwd=ROOT)
    atomic_json(out/'series-status.json',dict(phase=args.phase,status='complete',elapsed=time.monotonic()-started))


if __name__=='__main__':main()
