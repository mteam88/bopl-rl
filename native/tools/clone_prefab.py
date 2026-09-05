"""Pinned empty-player asset and C# Awake defaults for the native clone factory."""
import copy
import json
import struct
from functools import lru_cache
from actor_stream import ROOT
from verify_actor_spawn import queries as spawn_queries
from verify_actor_bounds import assets as boundary_assets
from verify_actors import ONE, vec


@lru_cache
def prefab():
    base = next(spawn_queries(725105))
    actor = copy.deepcopy(base['runtime']['actors'][0])
    source = json.loads((ROOT/'native/reference-export/assets.json').read_text())
    parts = {c['type']: c['properties'] for c in source['91b35711383743b4e9b73623dc72879b']['data']['components']}
    box = parts['DPhysicsBox']
    body = parts['PlayerBody']
    collision = parts['PlayerCollision']
    slime_asset = parts['SlimeController']
    actor.update(object=0, input=vec(), collision_hierarchy=box['hierarchyNumber'],
                 push_strength=slime_asset['pushOverlappingPlayersStr.m_rawValue'],
                 bite_threshold=slime_asset['scaleDiffBeforeNomNom.m_rawValue'])
    controller = actor['controller']
    controller.update(skip_update=False, apply_gravity=True, blocked=True)
    controller['physics'].update(scale=ONE, previous_scale=ONE, jumped=False,
                                 sync_rotation=True, cannot_unground=False)
    motion = controller['physics']['motion']
    motion.update(position=vec(), rotation=0, up=vec(0, 1), right=vec(1, 0),
                  velocity=vec(), external_velocity=vec(), grounded=False,
                  ground=None, local=0, grounded_speed=0)
    for key in ['position', 'velocity', 'external_velocity', 'previous_position',
                'previous_velocity', 'previous_imposed', 'previous_external', 'extra_swing_force']:
        actor['body'][key] = vec()
    actor['body'].update(rope=None, top=False, ignore_gravity=False,
                         rope_mass=body['RopePlayerMass01.m_rawValue'], holder=1,
                         physics=dict(grounded=False, jumped=False))
    actor['entity'] = dict(node=0, present=True, destroyed=False, active=True,
                           body_simulated=False, box=dict(scale=ONE,
                           min_scale=box['MinScale.m_rawValue'], max_scale=box['MaxScale.m_rawValue'],
                           minimum_extents=box['minimumExtents.m_rawValue'],
                           start_extents=dict(x=box['startExtents.x.m_rawValue'], y=box['startExtents.y.m_rawValue']),
                           initialized=False))
    trauma = struct.unpack('<f', struct.pack('<I', collision['deathTrauma']['f32']))[0]
    actor['collision'] = dict(invulnerable=0, was_killed=False, death_trauma=trauma)
    actor['bounds'] = copy.deepcopy(boundary_assets()[0])
    sample = dict(position=vec(), velocity=vec(), external_velocity=vec(), rotation=0, cooldowns=[0]*3)
    actor['rewind'] = dict(ring=[copy.deepcopy(sample) for _ in range(188)], head=0, count=0, rewinding=False)
    slime = actor['slime']
    slime['state'].update(mode=2, facing_right=True, was_upside_down=False, input=vec(), previous_input=0,
                          suppress=True, has_spawned=False, time_moonwalking=0, turn_input=0)
    slime['spawn'].update(physics_enabled=True, renderer_enabled=True, fast=False, next_ability=0,
                          abilityless=False, tutorial=False)
    slime['abilities'].update(slots=[], pool=[], timers=None, indicators=[], in_ability=False,
                              can_use=True, buttons=[False]*3)
    slime['animator'].update(current=None, next=None, body=None, callback=None, child_suffix='')
    slime['speed'] = dict(velocity=vec(), rotation=0)
    # The factory clears the sprite before PrepareToRevive, as the source does.
    slime['animator']['render'] = dict(sprite=None, flip_x=False, flip_y=False)
    shape = dict(kind='box', shape=dict(center=vec(), right=vec(), up=vec(), inverse_extents=vec()))
    physics_body = {k: 0 for k in ['inverse_mass', 'angular_velocity', 'orientation', 'inverse_inertia',
                    'bounciness', 'static_friction', 'dynamic_friction', 'gravity_scale', 'air_drag', 'rotational_drag']}
    physics_body.update(velocity=vec(), force=vec(), inverse_mass=-ONE)
    actor['pending_object'] = dict(id=0, hierarchy=6, collider=shape, body=physics_body,
        transform=dict(position=vec(), rotation=0, offset=vec(), right=vec(1, 0)), layer=8,
        active=True, callback=True, callback_destroyed=False, destroyed=False, composite=None,
        relative_pos=vec(), relative_orientation=0)
    nodes = [dict(parent=None, children=[1], active=True, present=True),
             dict(parent=0, children=[2, 3], active=True, present=True),
             dict(parent=1, children=[], active=True, present=True),
             dict(parent=1, children=[], active=True, present=True)]
    roles = ['physics', 'motor', 'collision', 'bounds', 'body', 'collider', 'rewind', 'animation', None]
    components = [dict(role=role, state=dict(owner=2 if role == 'animation' else 0,
                   kind='fix' if role is None else 'updatable', present=True, destroyed=False,
                   hierarchy=[1, 2, 3, 4, 5, 6, 8, 7, 0][i], enabled=True, ignore_activity=False))
                  for i, role in enumerate(roles)]
    return dict(actor=actor, nodes=nodes, components=components, registration=[4, 5, 2, 3, 0, 1, 6, 7])
