"""Extract the explicitly supported movement state from a canonical Unity trace.

This is a projection, not a substitute for full-state comparison. It preserves raw
fixed-point values and rejects missing fields instead of supplying guessed defaults.
"""
import argparse
import gzip
import json
from pathlib import Path


def fix(v): return v['$fix']
def vec(v): return {k:fix(v['Vec2.'+k]) for k in ('x','y')}
def number(ref): return int(ref['$unity'].split(':')[-1])
def fields(state, prefix): return {k.removeprefix(prefix+'.'):v for k,v in state.items() if k.startswith(prefix+'.')}

def project(state):
    objects={o['id']:o for o in state['updatables']}
    transforms={o['id']:fields(o['state'],'FixTransform') for o in state['transforms']}
    platforms=[]
    for shape in state['roundedRects']['items']:
        ident=int(shape['parent'].split(':')[1]);s=fields(objects[f'StickyRoundedRectangle:{ident}']['state'],'StickyRoundedRectangle')
        t=transforms[shape['parent']];r=fields(shape['collider'],'RoundedRect');b=fields(r['box'],'Box')
        platforms.append(dict(id=ident,surface=dict(center=vec(t['position']),right=vec(t['rightInner']),up=vec(t['upInner']),extents=vec(s['extents']),radius=fix(r['radius'])),
            collider=dict(inner={k:vec(b[v]) for k,v in [('center','center'),('right','right'),('up','up'),('inverse_extents','inverseExtents')]},radius=fix(r['radius'])),ice=s['platformType']['value']==1))
    players=[]
    for obj in state['updatables']:
        if obj['state']['$type']!='SlimeController':continue
        slime=fields(obj['state'],'SlimeController');p=fields(objects[slime['playerPhysics']['$unity']]['state'],'PlayerPhysics');body=fields(objects[p['body']['$unity']]['state'],'PlayerBody')
        constants={key:fix(p[name]) for key,name in dict(max_speed='maxSpeed',accel='accel',air_accel='airAccel',gravity_accel='gravity_accel',gravity_max_fall_speed='gravity_maxFallSpeed',gravity_modifier='gravity_modifier',jump_strength='jumpStrength',jump_extra_x='jumpExtraXStrength',jump_momentum='jumpKeptMomentum',jump_normal_scale='jumpNormalScaleFactor',jump_teleport='extraJumpTeleportMultiplier',friction='PlatformSlipperyness01',ice_friction='IcePlatformSlipperyness01').items()}
        ability=objects[slime['abilities'][0]['$unity']]
        dash_id=ability['state']['Ability.abilityComponents'][0]['$unity'];dash_obj=objects[dash_id];dash=fields(dash_obj['state'],'Dash')
        dp=fields(objects[dash['physics']['$unity']]['state'],'PlayerPhysics')
        saved_local=fix(p['<LocalPlatformPos>k__BackingField'])
        if dash_obj['active']:
            p=dp;body=fields(objects[dp['body']['$unity']]['state'],'PlayerBody')
        t=transforms[body['fixTransform']['$unity']]
        dash_state=dict(active=dash_obj['active'],time=fix(dash['timePassed']),updates=dash['updatesPassedInDash'],in_dash=dash['inDash'],input=vec(dash['inputVector']),hit_ground=number(dash['hitGround']) if dash['hitGround'] else None,hit_local=fix(dash['HitGroundLocalPosition']),saved_local=saved_local,local=fix(dp['<LocalPlatformPos>k__BackingField']))
        shape=next(x for x in state['boxes']['items'] if x['parent']==body['fixTransform']['$unity'])
        box=fields(shape['collider'],'Box')
        collider={key:vec(box[name]) for key,name in dict(center='center',right='right',up='up',inverse_extents='inverseExtents').items()}
        players.append(dict(alive=not obj['state']['MonoUpdatable.<IsDestroyed>k__BackingField'],id=slime['playerNumber'],position=vec(t['position']),rotation=fix(t['rotationInner']),up=vec(t['upInner']),velocity=vec(body['selfImposedVelocity']),external_velocity=vec(body['externalVelocity']),radius=fix(p['radius']),grounded=p['isGrounded'],ground=number(p['attachedGround']) if p['attachedGround'] else None,local=fix(p['<LocalPlatformPos>k__BackingField']),grounded_speed=fix(p['<groundedSpeed>k__BackingField']),previous_jump=False,constants=constants,cooldown=fix(slime['abilityCooldownTimers'][0]),dash=dash_state,collider=collider))
    return dict(tick=state['tick'],platforms=platforms,players=players)


def states(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:
        for line in f:
            row=json.loads(line)
            if row.get('kind')=='state':yield row['value']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace');p.add_argument('output');p.add_argument('--tick',type=int,default=240);args=p.parse_args()
    state=next(s for s in states(args.trace) if s['tick']==args.tick)
    Path(args.output).write_text(json.dumps(project(state),indent=2)+'\n')
