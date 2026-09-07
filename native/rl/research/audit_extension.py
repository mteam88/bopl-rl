"""Audit continuation state, immutable snapshots and all fixed evaluation jobs."""
import argparse
import json
from pathlib import Path
import torch
from continuation import digest
from evaluate_extension import checkpoint, jobs, plan
from run_extension import verify
from train import ROOT, atomic_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-final-review',action='store_true')
    args=parser.parse_args();out=args.output.resolve();protocol=read(out/'protocol.json')
    verify(out,protocol);records=[];budgets=set();libraries=set()
    incidents=read(out/'training-incidents.json');common=read(out/'fallback-checkpoint.json')['milestone']
    evaluation=read(out/'evaluation-status.json');assert evaluation['status']=='complete_with_omissions'
    assert read(out/'training-status.json')['failed']==['B-s73']
    for run in protocol['runs']:
        directory=out/'runs'/run['name'];config=read(directory/'config.json');finished=(directory/'result.json').exists()
        result=read(directory/('result.json' if finished else 'status.json'))
        if finished:assert result['status']=='complete' and result['learner_steps']>=protocol['total_learner_target']
        else:
            incident=next(i for i in incidents['incidents'] if i['run']==run['name'])
            assert run['name']=='B-s73' and incident['kind']=='observation_capacity'
            assert result['learner_steps']==incident['last_completed_learner_steps']==3841536
            assert '99 entities, 2 slimes' in (out/'B-s73.train.log').read_text()
        for key in ('workers','horizon','sequence','epochs','learning_rate','training_maps','episode_ticks'):
            assert config[key]==protocol[key],(run['name'],key)
        assert config['seed']==run['seed'] and config['no_eval'] and config['no_video']
        assert config['continuation']['checkpoint_sha256']==run['source_sha256']
        if run['condition']=='A':assert config['scripted_opponents'] is None
        else:assert config['scripted_manifest_sha256']==protocol['training_bots_sha256']
        for name,expected in config['source_sha256'].items():
            assert digest(directory/'source'/name)==expected,(run['name'],name)
        for name,expected in protocol['training_source_sha256'].items():
            assert config['source_sha256']['native/rl/research/'+name]==expected
        original=torch.load(ROOT/run['source_checkpoint'],map_location='cpu',weights_only=False)
        for field in ('obs_size','hidden','algorithm','gamma','gae_lambda','learning_rate','action_hz','observation_delay_ticks','own_previous_actions','loadout','stocks','pool_size','reward','minibatch_segments','pipeline_groups','cuda_graphs','torch_version','assembly_sha256','research_library_sha256','puffer_library_sha256'):
            assert config[field]==original['config'][field],(run['name'],field)
        restored=torch.load(directory/'checkpoints'/Path(run['source_checkpoint']).name,map_location='cpu',weights_only=False)
        for key in ('policy','optimizer','torch_rng','cuda_rng','step','learner_steps','updates'):
            torch.testing.assert_close(restored[key],original[key],rtol=0,atol=0)
        del original,restored
        milestones={}
        for milestone in (2000000,3000000,4000000):
            path=checkpoint(out,protocol,run,milestone)
            if not path.exists():
                assert not finished and milestone==4000000;continue
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert milestone<=saved['learner_steps']<milestone+1536
            milestones[milestone]=dict(learner_steps=saved['learner_steps'],sha256=digest(path))
        if finished:assert saved['step']==result['step'] and saved['updates']==result['updates']
        else:
            retained=sorted((directory/'checkpoints').glob('policy-*.pt'))[-1]
            saved=torch.load(retained,map_location='cpu',weights_only=False)
            assert saved['learner_steps']==3838464
            assert saved['learner_steps']<=result['learner_steps']
        metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics)==result.get('updates',result.get('update'))-run['source_updates']
        assert all(m['learner_fraction']==.75 for m in metrics)
        assert all(m['env'].get('game_errors',0)==0 and m['env'].get('observation_overflow',0)==0 for m in metrics)
        assert [m['learner_steps'] for m in metrics]==list(range(run['source_learner_steps']+1536,result['learner_steps']+1,1536))
        if finished:budgets.add((result['learner_steps'],result['step'],result['updates']))
        libraries.add(tuple(config[k] for k in ('assembly_sha256','research_library_sha256','puffer_library_sha256')))
        records.append(dict(name=run['name'],status='complete' if finished else 'interrupted',learner_steps=result['learner_steps'],new_learner_steps=result['learner_steps']-run['source_learner_steps'],source_snapshots=len(config['source_sha256']),milestones=milestones))
    assert len(budgets)==1 and len(libraries)==1
    assert sum(r['status']=='complete' for r in records)==5
    fallback=read(out/'fallback-checkpoint.json')
    assert fallback['source_sha256']=={r['name']:r['milestones'][common]['sha256'] for r in records}
    assert fallback['selection_protocol_sha256']==digest(out/'capacity-fallback-protocol.json')
    suites=[];replays=0
    expected_jobs=jobs(protocol)
    for a in protocol['crossplay']['seeds']:
        for b in protocol['crossplay']['seeds']:expected_jobs.append(dict(name=f'crossplay-{common}-A{a}-B{b}',kind='crossplay',milestone=common,a=a,b=b))
    for r in protocol['runs']:expected_jobs.append(dict(name=f"{r['name']}-{common}-transfer",kind='script',run=r['name'],milestone=common,suite='transfer'))
    assert len(expected_jobs)==len(evaluation['completed'])+len(evaluation['unavailable'])
    assert {j['name'] for j in expected_jobs}=={j['name'] for j in evaluation['completed']+evaluation['unavailable']}
    omitted={j['name']:j for j in evaluation['unavailable']}
    for job in expected_jobs:
        if job['name'] in omitted:
            assert job['milestone']==4000000 and (job.get('run')=='B-s73' or job.get('b')==73)
            assert not (out/'evaluations'/job['name']/'result.json').exists();continue
        directory=out/'evaluations'/job['name'];expected=plan(out,protocol,job);saved=read(directory/'schedule.json');result=read(directory/'result.json')
        # Raw paths describe the original workspace. Compare identity, not relocation.
        actual_tasks=saved['schedule']['tasks'] if 'schedule' in saved else saved['tasks']
        planned={t['name']:t for t in expected['tasks']}
        assert len(result['games'])==len(planned)==len(actual_tasks)
        assert {g['name'] for g in result['games']}==set(planned)
        for name,sha in saved['source_sha256'].items():assert digest(directory/'source'/name)==sha
        for name,sha in protocol['training_source_sha256'].items():assert saved['source_sha256'][name]==sha
        for game in result['games']:
            task=planned[game['name']]
            for key in ('map','side','seed'):assert task[key]==game[key]
            for key in ('learner','opponent'):
                for field,value in task[key].items():
                    if field!='path':assert game[key][field]==value
            assert read(directory/(game['name']+'.json'))==game
            if 'replay' in game:
                assert digest(directory/game['replay'])==game['replay_sha256'];replays+=1
        suites.append(dict(name=job['name'],games=len(result['games'])))
    traces=read(out/'failure-review/result.json')
    assert len(traces['cases'])==36
    for case in traces['cases']:
        assert digest(ROOT/case['original'])==case['sha256']
        assert case['identical_final_state_and_both_player_events']
        original=read(ROOT/case['original'])
        assert case['original_replay_hash_matched']==('replay_sha256' in original)
        if 'replay_sha256' in original:assert case['replay_sha256']==original['replay_sha256']
        directory=out/'failure-review'/(case['run']+'-'+case['case'])
        assert digest(directory/'game.rep')==case['replay_sha256']
        assert (directory/'frames.json.gz').is_file() and (directory/'decisions.npz').is_file()
    report=dict(schema=1,complete=False,available_work_verified=True,common_checkpoint=common,unavailable_evaluation_suites=evaluation['unavailable'],incidents=incidents,protocol_sha256=digest(out/'protocol.json'),runs=records,
                completed_run_budget=list(budgets)[0],restored_policy_optimizer_rng_counts_bit_exact=True,
                immutable_source_snapshots=True,binary_identities_match=True,evaluation_suites=suites,
                new_evaluation_games=sum(s['games'] for s in suites),verified_replays=replays,verified_failure_traces=36,original_failure_replays_matched=sum(c['original_replay_hash_matched'] for c in traces['cases']))
    if args.include_final_review:
        review=f'failure-review-{common//1000000}m'
        planned=read(out/(review+'-protocol.json'))
        final=read(out/review/'result.json')
        assert final['protocol_sha256']==digest(out/(review+'-protocol.json'))
        assert len(final['cases'])==36
        assert {(c['run'],c['case']) for c in final['cases']}=={(c['run'],c['case']) for c in planned['cases']}
        final_hashes={r['name']:r['milestones'][common]['sha256'] for r in records}
        for case in final['cases']:
            path=out/'evaluations'/(case['run']+f'-{common}-transfer')/(case['case']+'.json')
            original=read(path)
            assert digest(path)==case['sha256']
            assert case['checkpoint_sha256']==final_hashes[case['run']]==original['learner']['sha256']
            assert case['identical_final_state_and_both_player_events']
            assert case['original_replay_hash_matched']==('replay_sha256' in original)
            if 'replay_sha256' in original:assert case['replay_sha256']==original['replay_sha256']
            directory=out/review/(case['run']+'-'+case['case'])
            assert digest(directory/'game.rep')==case['replay_sha256']
            assert (directory/'frames.json.gz').is_file() and (directory/'decisions.npz').is_file()
        report['verified_matched_final_traces']=36
        report['original_final_replays_matched']=sum(c['original_replay_hash_matched'] for c in final['cases'])
    archive=out/'incidents/orchestration-A-s173'
    original_config=read(archive/'run/config.json')
    for name,sha in original_config['source_sha256'].items():assert digest(archive/'run/source'/name)==sha
    archived_suites=[]
    for file in (archive/'evaluations').glob('*/result.json'):
        source=read(file.parent/'schedule.json')
        for name,sha in source['source_sha256'].items():assert digest(file.parent/'source'/name)==sha
        games=read(file)['games']
        for game in games:
            assert read(file.parent/(game['name']+'.json'))==game
            if 'replay' in game:assert digest(file.parent/game['replay'])==game['replay_sha256']
        replacement=read(out/'evaluations'/file.parent.name/'result.json')['games']
        def identity_free(game):
            value=json.loads(json.dumps(game))
            for who in ('learner','opponent'):
                value[who].pop('path',None);value[who].pop('sha256',None)
            return value
        assert {g['name']:identity_free(g) for g in games}=={g['name']:identity_free(g) for g in replacement}
        archived_suites.append(dict(name=file.parent.name,games=len(games),replacement_outcomes_events_and_replays_identical=True))
    report['archived_operator_attempt_evaluations']=archived_suites
    comparison=read(out/'incidents/orchestration-A-s173/checkpoint-comparison.json')
    assert len(comparison)==2 and all(all(v['checks'].values()) for v in comparison.values())
    for filename,record in comparison.items():
        assert digest(out/'incidents/orchestration-A-s173/run/checkpoints'/filename)==record['original_sha256']
        assert digest(out/'runs/A-s173/checkpoints'/filename)==record['replacement_sha256']
    report['operator_retry_milestones_bit_exact']=comparison
    report['analysis_source_sha256']={name:digest(Path(__file__).parent/name) for name in ('audit_extension.py','plot_extension.py','trace_league_failures.py','analyze_league_failures.py','review_extension_final.py')}
    atomic_json(out/'audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
