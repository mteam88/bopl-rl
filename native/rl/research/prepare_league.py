"""Write the prospective three-condition protocol before training starts."""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
from scripted_opponents import load_specs, BotSpec
from train import atomic_json, ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if (out/'protocol.json').exists():raise ValueError('Protocol already frozen')
    base=load_specs(out/'train-bots.json')
    train=[variant for b in base for variant in
           (b,replace(b,name=b.name+'-patient',reaction=b.reaction+1,
                      aim_error=b.aim_error+.025,hold=b.hold+2,
                      aggression=max(.1,b.aggression-.2),distance=b.distance+2))]
    dev=[replace(b,name=b.name+'-dev',reaction=2,aim_error=.03,hold=17,
                 distance=b.distance+1,aggression=.6) for b in base]
    final=[replace(b,name=b.name+'-test',reaction=1,aim_error=.01,hold=15,
                   distance=b.distance-1,aggression=.85) for b in base]
    final.append(BotSpec('switcher-test','switcher',reaction=1,aim_error=.02,
                         distance=14,hold=16,lead=.8,aggression=.9))
    atomic_json(out/'training-bots.json',dict(schema=1,bots=[asdict(s) for s in train]))
    sources={name:digest(Path(__file__).parent/name) for name in
             ('train.py','selfplay.py','scripted_opponents.py','policy.py','ppo.py','vector.py',
              'inference.py','environment.py','telemetry.py','league_evaluate.py')}
    order=[('A',73),('B',173),('C',273),('B',73),('C',173),('A',273),('C',73),('A',173),('B',273)]
    protocol=dict(schema=1,description='Scripted opponents and Bow initialization: prospective 3x3 comparison',
                  conditions={'A':'untrained, current/historical opponents',
                              'B':'untrained, scripts/current/historical opponents',
                              'C':'Bow teacher initialization, same opponent mixture as B'},
                  runs=[dict(condition=c,seed=s,name=f'{c}-s{s}') for c,s in order],
                  learner_decisions=1000000,workers=8,horizon=128,sequence=32,epochs=3,
                  learning_rate=.0003,checkpoint_every_all_player_decisions=100000,
                  episode_ticks=3600,training_maps=[0,3,6],
                  scripted_fraction_of_fixed_opponent_assignments=2/3,
                  mixture_note='Half of workers current/current. Other half: 2/3 script, 1/3 historical, sampled at match boundaries. Actual decision exposure is measured.',
                  training_bots_sha256=digest(out/'training-bots.json'),
                  initialization_sha256=digest(out/'bow-initial.pt'),
                  initialization_note='Teacher-fit weights; exact Bow marginal, each other button starts with probability .001, all outputs trainable. Fresh Adam. Initialization is a bundled representation/action-prior intervention.',
                  initialization_cost_note='Shared existing 38-decision scripted example, 2000 supervised updates. Report original fit elapsed time separately. No Bow PPO run is used.',
                  development=dict(maps=[0,3,6],bots=[asdict(s) for s in dev],pairs_per_cell=8,seed_base=51001),
                  final=dict(maps=[1,2,5],bots=[asdict(s) for s in final],pairs_per_cell=16,seed_base=71001),
                  selection='Evaluate every final checkpoint. No hyperparameter or checkpoint selection from final results.',
                  metrics=['win fraction (draws not wins)','loss fraction','timeouts','kills','deaths','arrow hits','per-map/per-opponent scores','Bow retention'],
                  analysis='Report all three seeds, paired B-A and C-B differences, seed-pair bootstrap descriptive intervals, and worst opponent. Three training seeds limit inference. No claim of independent layouts from reset seeds.',
                  training_source_sha256=sources,
                  assembly_sha256=digest(ROOT/'Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll'),
                  research_library_sha256=digest(ROOT/'native/artifacts/rl/libbopl_research.so'))
    atomic_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(digest(out/'protocol.json')+'\n')
    print('PROTOCOL_FROZEN',digest(out/'protocol.json'))


if __name__=='__main__':main()
