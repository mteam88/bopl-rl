"""Component metadata and ordered contacts for the collision-selection oracle."""
import random
from copy import deepcopy

ONE=1<<32

def v(x=0,y=0):return dict(x=int(x*ONE),y=int(y*ONE))

def queries(count,seed):
    rng=random.Random(seed)
    for case in range(count):
        ids=list(range(1,case%6+2));player=rng.randrange(1,7)
        if case<120:ids=list(range(1,7))
        alive=(rng.randrange(8)!=0) if player in ids else None
        if case<120:alive=True
        globals=dict(players=ids,local_players=[i for i in ids if rng.randrange(2)],replay=bool(case%2),
            engine_achievement=case%7==0,arrow_achievement=case%7==1,rock_achievement=case%7==2)
        receiver=dict(player_id=player,player_alive=alive,invulnerable=0 if case%5 else rng.getrandbits(32),
            was_killed=bool(case%9==0),position=v(rng.choice([0,1,3,4]),0),right=[v(1,0),v(0,1),v(-1,0),v(.6,.8)][case%4],squish_offset=v(rng.choice([0,.5,-.5]),[-.5,0,.5][case%3]),
            squishable=case%13!=0,physics=dict(grounded=case%11!=0,attached=dict(body=1,destroyed=case%17==0)),
            ignored=[None]*6,ignored_beam=rng.choice([-1,0,17,2147483647]))
        if case%23==0:receiver['physics']=None
        elif case%19==0:receiver['physics']['attached']=None
        hits=[]
        for i,layer in enumerate([11,11,16,18,9,8,8,26,0,18]):
            owner=rng.choice([player,*ids,1000])
            box=dict(center=v(),right=v(3,0),up=v(0,3),inverse_extents=v(1/3,1/3))
            shape=[dict(kind='circle',shape=dict(center=v(),radius=3*ONE)),dict(kind='box',shape=box),dict(kind='rounded_rect',shape=dict(inner=box,radius=ONE//2))][(i+case)%3]
            h=dict(id=i+10,layer=layer,collision_layer=layer,hierarchy=i+10,transform_present=True,destroyed=False,collider=shape,
                platform=None,orb_kind=None,explosion=None,hitbox_owner=None,projectile=None,holder_owner=None,ability=None,parent_ability=None,chain_lightning=False)
            if layer==11 or layer==0:
                h['collision_layer']=11
                h['platform']=dict(body=1 if case%10==0 else 2,pickup=i==0,engine=dict(owner=owner,destroyed=case%8==0,on=case%4!=0))
            if layer in (16,18):
                if i==2 and case%3==0:h['hitbox_owner']=owner
                else:h['explosion']=dict(owner=owner,hurt_owner=bool(rng.randrange(2)),beam=i==3,beam_id=rng.choice([None,17,-17,0,2147483647,-2147483648]))
                h['chain_lightning']=i==9
            if layer in (9,26):
                if layer==26:h['orb_kind']=case%6
                if case%3:h['projectile']=dict(owner=owner,hurt_owner=bool(rng.randrange(2)),age=rng.choice([0,5*ONE//2-1,5*ONE//2,5*ONE]),arrow_destroyed=rng.choice([None,True,False]))
                else:h['holder_owner']=owner
            if layer==8:
                a=dict(owner=owner,kills_on_contact=case%4!=0,rock=dict(hits=rng.choice([0,1,2,2147483647]),player_local=rng.choice([None,True,False])))
                h['ability' if i==5 else 'parent_ability']=a
            if (i==3 and case%5==1) or (i==5 and case%5==2):
                h['collision_layer']=11;h['platform']=dict(body=2,pickup=False,engine=None)
            if i==0 and case%5==3:
                h['layer']=9;h['holder_owner']=owner
            if case%31==0 and i==7:h['orb_kind']=None
            if case%37==0 and i==2:h['explosion']=None;h['hitbox_owner']=None
            if case%41==0 and i==4:h['projectile']=None;h['holder_owner']=None
            if case%12==0 and i%3==0:h['destroyed']=True
            if case%14==0 and i==5:h['transform_present']=False
            if case%16==0 and i==0:h['hierarchy']=0
            hits.append(h)
        if case%6==0:receiver['ignored'][(case//6)%6]=dict(hierarchy=hits[2]['hierarchy'],destroyed=case%12==0)
        actions=[dict(op='collide',hit=i,callback=False,kill_result=bool((case+i)%3)) for i in range(len(hits))]
        actions+= [dict(op='collide',hit=5,callback=False,kill_result=True)]*3
        actions+=[dict(op='update'),dict(op='collide',hit=2,callback=True,kill_result=False),dict(op='collide',hit=3,callback=True,kill_result=True),
            dict(op='update'),dict(op='invulnerable',enabled=True,mask=3),dict(op='collide',hit=2,callback=False,kill_result=True),
            dict(op='invulnerable',enabled=False,mask=1),dict(op='collide',hit=2,callback=False,kill_result=True),dict(op='invulnerable',enabled=False,mask=0xffffffff),
            dict(op='copy_ignored',ignored=[dict(hierarchy=12,destroyed=False)]+[None]*5),dict(op='collide',hit=2,callback=False,kill_result=True),
            dict(op='copy_ignored',ignored=[None]*6),dict(op='ignore_beam',id=17),dict(op='collide',hit=3,callback=False,kill_result=True)]
        yield dict(operation='dynamics_player_collision',receiver=receiver,globals=globals,hits=hits,actions=deepcopy(actions))
