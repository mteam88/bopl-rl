"""Continue the six fixed A/B policies to four million learner decisions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from train import ROOT, atomic_json
from continuation import digest


def verify(out,protocol):
    assert digest(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for name,expected in protocol['training_source_sha256'].items():
        assert digest(Path(__file__).parent/name)==expected,name
    assert digest(out/'training-bots.json')==protocol['training_bots_sha256']
    assert digest(ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll')==protocol['assembly_sha256']
    assert digest(ROOT/'native/artifacts/rl/libbopl_research.so')==protocol['research_library_sha256']
    for run in protocol['runs']:
        assert digest(ROOT/run['source_checkpoint'])==run['source_sha256']
        assert digest(out/(run['name']+'-continuation.json'))==run['context_sha256']


def training_command(out,protocol,run):
    directory=out/'runs'/run['name']
    command=[sys.executable,str(Path(__file__).with_name('train.py')),
             '--output',str(directory),'--seed',str(run['seed']),
             '--load',str(ROOT/run['source_checkpoint']),
             '--continuation-context',str(out/(run['name']+'-continuation.json')),
             '--learner-steps',str(protocol['total_learner_target']),
             '--workers',str(protocol['workers']),'--horizon',str(protocol['horizon']),
             '--sequence',str(protocol['sequence']),'--epochs',str(protocol['epochs']),
             '--lr',str(protocol['learning_rate']),'--episode-ticks',str(protocol['episode_ticks']),
             '--checkpoint-every',str(protocol['checkpoint_every']),
             '--eval-at',','.join(map(str,protocol['intermediate_all_player_thresholds'])),
             '--training-maps',','.join(map(str,protocol['training_maps'])),'--no-eval','--no-video']
    if run['condition']=='B':command+=['--scripted-opponents',str(out/'training-bots.json'),
                                      '--scripted-fraction',str(protocol['scripted_fraction'])]
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();protocol=json.loads((out/'protocol.json').read_text())
    start=time.monotonic();completed=[]
    try:
        verify(out,protocol)
        for run in protocol['runs']:
            verify(out,protocol);directory=out/'runs'/run['name']
            if not (directory/'result.json').exists():
                if directory.exists():raise ValueError(f'Incomplete continuation needs inspection: {directory}')
                directory.parent.mkdir(exist_ok=True)
                command=training_command(out,protocol,run)
                atomic_json(out/'training-status.json',dict(status='running',run=run['name'],completed=completed,elapsed=time.monotonic()-start))
                with (out/(run['name']+'.train.log')).open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
            result=json.loads((directory/'result.json').read_text())
            assert result['status']=='complete' and result['learner_steps']>=protocol['total_learner_target']
            completed.append(run['name']);print('TRAINING_COMPLETE',run['name'],result['learner_steps'],flush=True)
        atomic_json(out/'training-status.json',dict(status='complete',completed=completed,elapsed=time.monotonic()-start))
    except BaseException as error:
        atomic_json(out/'training-status.json',dict(status='failed',error=str(error),completed=completed,elapsed=time.monotonic()-start));raise


if __name__=='__main__':main()
