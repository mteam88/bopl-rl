"""Verify completed-run identities, equal budgets and paired initial weights."""
import argparse
import json
from pathlib import Path
import torch
from prepare_league import digest
from run_league import verify
from train import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    protocol=json.loads((out/'protocol.json').read_text());verify(out,protocol)
    records=[];initial={};budgets=set();libraries=set()
    for run in protocol['runs']:
        directory=out/'runs'/run['name']
        result=json.loads((directory/'result.json').read_text())
        config=json.loads((directory/'config.json').read_text())
        assert result['status']=='complete'
        assert config['seed']==run['seed'] and config['learning_rate']==protocol['learning_rate']
        assert config['training_maps']==protocol['training_maps']
        budgets.add((result['learner_steps'],result['step'],result['updates']))
        libraries.add((config['puffer_library_sha256'],config['research_library_sha256'],config['assembly_sha256']))
        if run['condition']=='A':assert config['scripted_opponents'] is None
        else:assert config['scripted_manifest_sha256']==protocol['training_bots_sha256']
        if run['condition']=='C':assert config['initialize_sha256']==protocol['initialization_sha256']
        else:assert config['initialize'] is None
        for relative,expected in config['source_sha256'].items():
            assert digest(directory/'source'/relative)==expected,(run['name'],relative)
        checkpoint=directory/'checkpoints'/Path(result['checkpoint']).name
        checkpoint_hash=digest(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['learner_steps']==result['learner_steps']
        initial[run['condition'],run['seed']]=torch.load(directory/'checkpoints/policy-000000000.pt',map_location='cpu',weights_only=False)['policy']
        evaluation_counts={}
        for suite in ('development','final'):
            evaluation_dir=out/'evaluations'/run['name']/suite
            evaluated=json.loads((evaluation_dir/'result.json').read_text())
            plan=json.loads((evaluation_dir/'schedule.json').read_text())
            for filename,expected in plan['source_sha256'].items():
                assert digest(evaluation_dir/'source'/filename)==expected
            for filename,expected in protocol['training_source_sha256'].items():
                assert plan['source_sha256'][filename]==expected
            for game in evaluated['games']:
                assert game['learner']['sha256']==checkpoint_hash
                assert json.loads((evaluation_dir/(game['name']+'.json')).read_text())==game
                if 'replay' in game:
                    assert digest(evaluation_dir/game['replay'])==game['replay_sha256']
            spec=protocol[suite]
            expected=len(spec['maps'])*len(spec['bots'])*spec['pairs_per_cell']*2
            assert len(evaluated['games'])==expected
            evaluation_counts[suite]=expected
        retention=json.loads((out/'retention'/run['name']/'result.json').read_text())
        assert retention['checkpoint_sha256']==checkpoint_hash and retention['trials']==32
        records.append(dict(name=run['name'],learner_steps=result['learner_steps'],
                            all_player_steps=result['step'],source_files=len(config['source_sha256']),
                            checkpoint_sha256=checkpoint_hash,evaluation_counts=evaluation_counts))
    assert len(budgets)==1 and len(libraries)==1
    seeds=sorted({r['seed'] for r in protocol['runs']})
    for seed in seeds:
        for key in initial['A',seed]:
            torch.testing.assert_close(initial['A',seed][key],initial['B',seed][key],rtol=0,atol=0)
    source=torch.load(out/'bow-initial.pt',map_location='cpu',weights_only=False)['policy']
    for seed in seeds:
        for key in source:torch.testing.assert_close(initial['C',seed][key],source[key],rtol=0,atol=0)
    report=dict(schema=1,complete=True,runs=records,equal_budgets=list(budgets)[0],
                paired_A_B_initial_weights_bit_exact=True,all_C_initial_weights_bit_exact=True,
                source_snapshots_match=True,binary_identities_match=True,protocol_sha256=digest(out/'protocol.json'))
    atomic_json(out/'audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
