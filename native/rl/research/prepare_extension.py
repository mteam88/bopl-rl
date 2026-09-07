"""Freeze cross-play, continuation and failure-review selections before extending."""
import argparse
import json
from pathlib import Path
import shutil
from train import ROOT, atomic_json
from continuation import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Protocol already frozen')
    old=ROOT/'native/artifacts/rl/opponent-league'
    source=json.loads((old/'protocol.json').read_text())
    # The continuation branch is the only change to the old training mechanism.
    for name,expected in source['training_source_sha256'].items():
        if name!='train.py' and digest(Path(__file__).parent/name)!=expected:
            raise ValueError(f'Prior training/evaluation source changed: {name}')
    shutil.copyfile(old/'training-bots.json',out/'training-bots.json')
    order=[('A',73),('B',173),('A',273),('B',73),('A',173),('B',273)]
    runs=[]
    for condition,seed in order:
        name=f'{condition}-s{seed}';directory=old/'runs'/name
        result=json.loads((directory/'result.json').read_text())
        checkpoint=directory/'checkpoints'/Path(result['checkpoint']).name
        metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        recent=metrics[-1]['pool_steps']
        historical=[dict(step=step,path=f'policy-{step:09d}.pt',sha256=digest(directory/'checkpoints'/f'policy-{step:09d}.pt')) for step in recent]
        context=dict(schema=1,checkpoint_sha256=digest(checkpoint),historical=historical,
                     checkpoint_every=source['checkpoint_every_all_player_decisions'],
                     next_checkpoint=recent[-1]+source['checkpoint_every_all_player_decisions'],
                     reset_note='Fresh games, zero recurrent memories and private NumPy RNG from original seed. Restore Torch RNG, weights, Adam, counts, recent historical pool and historical snapshot cadence. Initial fixed historical opponents use the latest restored historical snapshot.')
        atomic_json(out/(name+'-continuation.json'),context)
        runs.append(dict(name=name,condition=condition,seed=seed,
                         source_checkpoint=str(checkpoint.relative_to(ROOT)),source_sha256=digest(checkpoint),
                         source_learner_steps=result['learner_steps'],source_all_player_steps=result['step'],
                         source_updates=result['updates'],context_sha256=digest(out/(name+'-continuation.json'))))
    selections=[]
    for seed in (73,173,273):
        games=json.loads((old/'evaluations'/f'B-s{seed}'/'final/result.json').read_text())['games']
        for style in ('archer-test','evasive-test'):
            failures=[g for g in games if g['opponent']['spec']['name']==style and g['loss'] and not g['win']]
            failure=sorted(failures,key=lambda g:(g['map'],g['seed'],g['side']))[0]
            for c in 'AB':
                original=old/'evaluations'/f'{c}-s{seed}'/'final'/(failure['name']+'.json')
                selections.append(dict(run=f'{c}-s{seed}',case=failure['name'],original=str(original.relative_to(ROOT)),
                                       sha256=digest(original),selection='First B loss by map, action-seed, side for each seed and ranged style; matched A case regardless of its outcome'))
    names=list(source['training_source_sha256'])+['continuation.py']
    protocol=dict(schema=1,description='Bounded A/B continuation and direct cross-play',
                  parent_protocol_sha256=digest(old/'protocol.json'),runs=runs,
                  total_learner_target=4000000,checkpoints=[1000000,2000000,3000000,4000000],
                  intermediate_all_player_thresholds=[2666667,4000000],
                  workers=8,horizon=128,sequence=32,epochs=3,learning_rate=.0003,
                  checkpoint_every=100000,training_maps=source['training_maps'],episode_ticks=3600,
                  scripted_fraction=source['scripted_fraction_of_fixed_opponent_assignments'],
                  training_bots_sha256=digest(out/'training-bots.json'),
                  development=source['development'],transfer=source['final'],
                  evaluation_note='Previously inspected script benchmark is now fixed tracking evidence. Evaluate development at 1/2/3/4M and transfer at 1/4M. Reuse all original 1M script games. No tuning or checkpoint selection; report every milestone.',
                  crossplay=dict(maps=[0,1,2,3,5,6],seeds=[73,173,273],pairs_per_cell=8,
                                 seed_base=910001,milestones=[1000000,4000000],
                                 note='All nine A/B training-seed combinations, both player sides. A is learner field, B opponent field. Same game and per-player action seeds at both milestones.'),
                  failure_review=selections,training_source_sha256={name:digest(Path(__file__).parent/name) for name in names},
                  assembly_sha256=source['assembly_sha256'],research_library_sha256=source['research_library_sha256'],
                  decision_rule='Compare A and B at fixed budgets; report script and cross-play outcomes separately, with ties/draws excluded from wins. No curriculum changes, new abilities or hyperparameter tuning.',
                  outcome_note='Shared WonThisRound flags on simultaneous deaths count as draws in analysis. Existing training reward remains unchanged across both conditions.')
    atomic_json(out/'protocol.json',protocol);(out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    print('EXTENSION_PROTOCOL_FROZEN',digest(out/'protocol.json'))


if __name__=='__main__':main()
