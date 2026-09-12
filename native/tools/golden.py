"""Export auditable Unity projections, then verify them through the real C ABI.

The compared fields are movement, dash clocks, cached colliders, platform
geometry, and alive masks. Presentation, unrelated abilities, and post-death
match progression are outside this projection.
"""
import argparse,copy,gzip,hashlib,json,sys
from pathlib import Path
from project_state import project
from reference import ROOT,read_trace,first_difference
sys.path.insert(0,str(ROOT/'native/python'))
from bopl_native import World

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def comparable(state):
    state=copy.deepcopy(state)
    state['players']=[p for p in state['players'] if p['alive']]
    for p in state['players']:
        p.pop('previous_jump') # Checked via subsequent jump behavior; not in Unity projection.
        p['dash'].pop('saved_local') # Saved only at entry in Rust, original body persists in Unity.
        # Rust-side bookkeeping the Unity projection has no field for (listed
        # explicitly so a new field still has to be classified here). The
        # collider the first two cache is compared through 'collider' itself;
        # the jump/gravity flags through the resulting movement; the dash exit
        # rotation only feeds the RL entity table.
        for key in ('collider_rotation','collider_basis','jumped_this_frame','skip_next_gravity'):p.pop(key,None)
        p['dash'].pop('exit_rotation',None)
    return state

def actions(value):
    result=[]
    for p in (value if isinstance(value,list) else [value['InputPacketQuad.'+player] for player in ('p1','p2')]):
        for key in ('ab2','ab3','w','a','s','d','start','select'):
            if p['InputPacket.'+key]:raise ValueError('unsupported input in movement golden: '+key)
        result.append([p['InputPacket.joystickAngle'],int(p['InputPacket.jump'])+2*int(p['InputPacket.ab1'])])
    return result

def export(args):
    provenance=json.loads(args.trace.with_suffix('.provenance.json').read_text())
    if digest(args.trace)!=provenance['trace_sha256']:raise ValueError('trace hash mismatch')
    records=[];initial=None;finished=False;input_value=None;players=2;setup=None
    # Consume the complete validated trace even after the episode ends.
    for row in read_trace(args.trace):
        if row['kind']=='header':
            setup=row.get('setup')
            if setup:
                players=len(setup['players'])
                if setup['level']!=0 or setup['stocks']!=0 or any(p['abilities']!=[2] or p['keyboard'] for p in setup['players']):raise ValueError('unsupported movement setup')
        elif row['kind']=='input':input_value=row['value']
        elif row['kind']=='state':
            if row['value']['tick']==240:initial=project(row['value'])
            elif row['value']['tick']>240 and not finished:
                expected=project(row['value']);records.append(dict(actions=actions(input_value),expected=expected))
                finished=len(expected['players'])!=players or not all(p['alive'] for p in expected['players'])
    if initial is None or not records:raise ValueError('missing post-spawn episode')
    with World(players=players) as world:
        diff=first_difference(comparable(initial),comparable(world.state()))
        if diff:raise ValueError('reset differs: '+str(diff))
    reset_path=ROOT/('native/data/level1-movement.json' if players==2 else f'native/data/level1-movement-{players}.json')
    header=dict(schema='bopl-movement-golden-v1',players=players,reference=json.loads((ROOT/'native/data/reference.json').read_text())['reference_commit'],
        scope=__doc__,source_trace_sha256=provenance['trace_sha256'],replay_sha256=provenance['replay_sha256'],
        game_assembly_sha256=provenance['game_assembly_sha256'],reset_sha256=digest(reset_path),
        steps=len(records),terminal=finished)
    data=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in [header,*records]).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(gzip.compress(data,mtime=0))
    print(json.dumps(dict(exported=str(args.output),steps=len(records),terminal=finished)))

def verify(paths):
    if not paths:raise ValueError('no golden fixtures found')
    report=[]
    for path in paths:
        with gzip.open(path,'rt') as f:
            header=json.loads(next(f));records=[json.loads(l) for l in f]
        if header['schema']!='bopl-movement-golden-v1' or header['steps']!=len(records) or not records:raise ValueError('invalid golden')
        players=header.get('players',2)
        reset_path=ROOT/('native/data/level1-movement.json' if players==2 else f'native/data/level1-movement-{players}.json')
        if header['reset_sha256']!=digest(reset_path):raise ValueError('reset changed')
        with World(players=players) as world:
            for index,record in enumerate(records):
                world.step(record['actions']);actual=world.state();expected=record['expected']
                diff=first_difference(comparable(expected),comparable(actual))
                if diff:raise ValueError(f'{path.name} tick {actual["tick"]}: {diff}')
            if world.terminal!=header['terminal']:raise ValueError('terminal mismatch')
        report.append(dict(fixture=path.name,steps=len(records),terminal=header['terminal'],match=True,sha256=digest(path)))
    print(json.dumps(dict(match=True,steps=sum(r['steps'] for r in report),fixtures=report),indent=2))
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    exp=sub.add_parser('export');exp.add_argument('trace',type=Path);exp.add_argument('output',type=Path)
    check=sub.add_parser('verify');check.add_argument('paths',nargs='*',type=Path)
    args=parser.parse_args()
    if args.command=='export':export(args)
    else:verify(args.paths or sorted((ROOT/'native/fixtures/reference').glob('*.jsonl.gz')))
