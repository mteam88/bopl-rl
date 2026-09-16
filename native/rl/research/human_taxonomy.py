"""Loss taxonomy of the human-vs-agent rounds (Matthew on the Mac vs checkpoints/bots).

    python human_taxonomy.py facts    --extracted DIR --out facts.json     # per-round terminal facts from the replayed frames
    python human_taxonomy.py tables   --facts facts.json [--sim-facts unity_facts.json] --out tables.md   # markdown tables for the report

Frames come from human_replays.py (Unity Linux build replaying the recorded inputs, one
frame per level tick). Player 0 is the human, player 1 the agent. The taxonomy is built
from the facts, not the other way around: `facts` records everything mechanical about
the death (cause, killer, arrow geometry, victim state, pre-death window statistics) and
`classify` maps those to named patterns.
"""
import argparse
from collections import Counter, defaultdict
import glob
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from human_replays import load_game  # noqa: E402

DASH, BOW, PLATFORM = 2, 4, 14
ABILITY_NAMES = {0: 'NullAbility', 1: 'Random', 2: 'Dash', 3: 'Grenade', 4: 'Bow', 5: 'Engine', 6: 'Blink gun', 7: 'Gust', 8: 'Growth ray', 9: 'Rock',
                 10: 'Missile', 11: 'Spike', 12: 'Time stop', 13: 'Smoke', 14: 'Platform', 15: 'Revival', 16: 'Roll', 17: 'Shrink ray', 18: 'Black hole',
                 19: 'Invisibility', 20: 'Meteor', 21: 'Throw', 22: 'Push', 23: 'Tesla coil', 24: 'Mine', 25: 'Teleport', 26: 'Drill', 27: 'Grappling hook',
                 28: 'Beam', 29: 'Duplicator', 30: 'Magnet gun', 31: 'Rewind', 32: 'Web swing', 33: 'Poke ball', 34: 'Gold mushroom', 35: 'Tack',
                 36: 'Freeze ray', 37: 'Mjolnir', 38: 'Portal gun'}
STANDARD = {DASH, BOW, PLATFORM}
WARMUP = 240
STATE_FALL = 2          # SlimeController.State.fall: airborne
CAUSE = {0: 'other', 1: 'squished', 2: 'accident', 3: 'eaten', 4: 'beamed', 5: 'alive'}
# Agent strength tiers (Matthew's record, brief): strong = o7-041/o7-016/policy-1333362688 and the v2-hard bots.
TIERS = {
    'strong-checkpoint': {'o7-041-final.bin', 'o7-016-final.bin', 'policy-1333362688.bin'},
    'mid-checkpoint': {'o7-000-final.bin', 'o7-007-final.bin', 'o7-033-final.bin', 'o7-035-final.bin', 'o3-032-final.bin', 'o3-017-final.bin', 'o3-019-final.bin', 'o3-012-final.bin'},
    'weak-checkpoint': {'o3-001-final.bin', 'o3-002-final.bin', 'o3-009-final.bin', 'run007-final.bin'},
}


def agent_name(row):
    return row['agent'].split('/')[-1] if row['b'].startswith('policy:') else row['b'][4:]


def tier(row):
    name = agent_name(row)
    if row['b'].startswith('bot:'):
        return 'bot'
    for t, names in TIERS.items():
        if name in names:
            return t
    return 'other-checkpoint'


def stratum(row):
    human = set(row['human_abilities'][0])
    if human != STANDARD:
        return 'C'
    if row['map'] in (0, 1, 2):
        return 'A'
    if row['map'] in (3, 4):
        return 'D'
    return 'B'


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def slime_of(frame, player):
    """The slime record of player index `player` (player ids are 1-based) or None when dead/absent."""
    pid = frame['players'][player]['id']
    for s in frame['slimes']:
        if s['owner'] == pid:
            return s
    return None


def ability_state(slime, ability):
    """(present, ready, active, charge) for the ability with catalog id `ability` (slot order differs per player)."""
    if slime is None or ability not in slime['abilities']:
        return None
    i = slime['abilities'].index(ability)
    f = slime['ability_flags'][i]
    return dict(ready=bool(f & 2), active=bool(f & 8), charge=(f >> 4) & 7, fired=bool(f & 0x80),
                timer=slime['timers'][i], cooldown=slime['cooldowns'][i], remaining=max(0.0, slime['cooldowns'][i] - slime['timers'][i]))


def active_ability(slime):
    if slime is None:
        return None
    for a, f in zip(slime['abilities'], slime['ability_flags']):
        if f & 8:
            return a
    return None


def pos(frame, player):
    s = slime_of(frame, player)
    if s is not None:
        return (s['x'], s['y'])
    p = frame['players'][player]
    return (p['x'], p['y'])


def event_ticks(frames, player, field, lo=0, hi=None):
    """Level ticks at which a cumulative telemetry counter incremented."""
    out = []
    hi = len(frames) if hi is None else hi
    last = frames[lo]['telemetry'][player][field]
    for f in frames[lo + 1:hi]:
        v = f['telemetry'][player][field]
        if v > last:
            out.extend([f['t']] * (v - last))
        last = v
    return out


def button_presses(frames, player, bit, lo=0, hi=None):
    """Level ticks where the player's input bit went 0->1 (inputs[t] is the packet applied to reach frame t)."""
    hi = len(frames) if hi is None else hi
    out = []
    prev = 0
    for f in frames[lo:hi]:
        b = f['inputs'][player]['buttons'] if 'inputs' in f else 0
        if b & bit and not prev & bit:
            out.append(f['t'])
        prev = b
    return out


def platform_owners(frames):
    """Spawned platform id -> (owner player index, spawn tick, spawn xy). A PlatformTransform morph spawns its
    StickyRect at the morph body's position; the nearest player with the platform ability active owns it."""
    seen = {e['id'] for e in frames[0]['platforms']} | {e['id'] for e in frames[WARMUP]['platforms']}
    owners = {}
    for f in frames[WARMUP:]:
        for e in f['platforms']:
            if e['id'] in seen:
                continue
            seen.add(e['id'])
            best = None
            for i in range(len(f['players'])):
                s = slime_of(f, i)
                if s is None:
                    continue
                st = ability_state(s, PLATFORM)
                if st and st['active']:
                    d = dist((s['x'], s['y']), (e['x'], e['y']))
                    if best is None or d < best[0]:
                        best = (d, i)
            owners[e['id']] = dict(owner=(best[1] if best and best[0] < 6.0 else None), tick=f['t'], x=e['x'], y=e['y'], distance=(best[0] if best else None))
    return owners


def platform_sdf(e, x, y):
    px, py = x - e['x'], y - e['y']
    fex, fey = math.hypot(e['rx'], e['ry']), math.hypot(e['ux'], e['uy'])
    rx, ry = (e['rx'] / fex, e['ry'] / fex) if fex > 1e-6 else (1.0, 0.0)
    ux, uy = (e['ux'] / fey, e['uy'] / fey) if fey > 1e-6 else (-ry, rx)
    lx, ly = px * rx + py * ry, px * ux + py * uy
    qx, qy = abs(lx) - fex, abs(ly) - fey
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - e['radius']


def ground_level(frames):
    """Lowest platform surface y on the level (proxy for the water line on levels the sim does not know)."""
    ys = []
    for e in frames[WARMUP]['platforms']:
        ys.append(e['y'] - math.hypot(e['ux'], e['uy']) - e['radius'])
    return min(ys) if ys else None


def window_stats(frames, idx, player, other, window=120):
    """Statistics over the `window` ticks before frame index idx for `player` (the eventual victim or the agent)."""
    lo = max(WARMUP, idx - window)
    seg = frames[lo:idx + 1]
    alive = [f for f in seg if f['players'][player]['flags'] & 1 and f['players'][other]['flags'] & 1]
    if not alive:
        return {}
    d = [dist(pos(f, player), pos(f, other)) for f in alive]
    airborne = [slime_of(f, player) is not None and slime_of(f, player)['state'] == STATE_FALL for f in alive]
    s_end = slime_of(alive[-1], player)
    dash = ability_state(s_end, DASH)
    plat = ability_state(s_end, PLATFORM)
    bow = ability_state(s_end, BOW)
    arrows_p = event_ticks(frames, player, 'arrows_fired', lo, idx + 1)
    arrows_o = event_ticks(frames, other, 'arrows_fired', lo, idx + 1)
    all_arrows_p = event_ticks(frames, player, 'arrows_fired', WARMUP, idx + 1)
    dashes_p = event_ticks(frames, player, 'dash_activations', WARMUP, idx + 1)
    t_end = alive[-1]['t']
    vel = [math.hypot(slime_of(f, player)['vx'], slime_of(f, player)['vy']) for f in alive if slime_of(f, player)]
    return dict(distance_mean=sum(d) / len(d), distance_end=d[-1], distance_min=min(d), airborne_fraction=sum(airborne) / len(airborne),
                airborne_end=airborne[-1], dash_ready_end=bool(dash and dash['ready']), dash_remaining_end=dash['remaining'] if dash else None,
                platform_ready_end=bool(plat and plat['ready']), bow_ready_end=bool(bow and bow['ready']),
                arrows_fired_window=len(arrows_p), opponent_arrows_window=len(arrows_o),
                ticks_since_last_shot=(t_end - all_arrows_p[-1]) if all_arrows_p else None,
                ticks_since_last_dash=(t_end - dashes_p[-1]) if dashes_p else None,
                speed_mean=(sum(vel) / len(vel)) if vel else None, y_end=pos(alive[-1], player)[1], y_other_end=pos(alive[-1], other)[1],
                height_advantage_end=pos(alive[-1], player)[1] - pos(alive[-1], other)[1])


def arrow_birth(frames, arrow_id, idx):
    for j in range(idx, WARMUP - 1, -1):
        if not any(a['id'] == arrow_id for a in frames[j]['arrows']):
            return j + 1
    return WARMUP


def death_facts(frames, dead, owners):
    """Everything mechanical about the death of player `dead` (0 human, 1 agent)."""
    killer = 1 - dead
    idx = next(i for i, f in enumerate(frames) if not f['players'][dead]['flags'] & 1 and i >= WARMUP)
    at, before = frames[idx], frames[idx - 1]
    p = at['players'][dead]
    tel_before, tel_at = before['telemetry'][dead], at['telemetry'][dead]
    kind = next((k for k in ('deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other') if tel_at[k] > tel_before[k]), 'unknown')
    sb = slime_of(before, dead)
    sk = slime_of(before, killer)
    out = dict(dead=dead, tick=at['t'], episode_tick=at['t'] - WARMUP, cause=CAUSE.get(p['cause_of_death'], p['cause_of_death']), killer_kind=kind[7:],
               x=p['x'], y=p['y'], victim_state=sb['state'] if sb else None, victim_airborne=bool(sb and sb['state'] == STATE_FALL),
               victim_active_ability=ABILITY_NAMES.get(active_ability(sb)) if sb else None,
               killer_active_ability=ABILITY_NAMES.get(active_ability(sk)) if sk else None,
               killer_airborne=bool(sk and sk['state'] == STATE_FALL),
               distance_at_death=dist(pos(before, dead), pos(before, killer)) if sk else None,
               victim_dash=ability_state(sb, DASH), victim_platform=ability_state(sb, PLATFORM), victim_bow=ability_state(sb, BOW),
               killer_dash=ability_state(sk, DASH), killer_platform=ability_state(sk, PLATFORM), killer_bow=ability_state(sk, BOW),
               victim_velocity=(sb['vx'], sb['vy']) if sb else None, killer_velocity=(sk['vx'], sk['vy']) if sk else None,
               ground_level=ground_level(frames))
    out['victim_window'] = window_stats(frames, idx - 1, dead, killer)
    out['killer_window'] = window_stats(frames, idx - 1, killer, dead)
    # Last dash / platform / arrow of each side before the death.
    for who, name in ((dead, 'victim'), (killer, 'killer')):
        for field, label in (('dash_activations', 'dash'), ('platform_activations', 'platform'), ('arrows_fired', 'arrow'), ('bow_activations', 'bow')):
            ev = event_ticks(frames, who, field, WARMUP, idx + 1)
            out[f'{name}_last_{label}_age'] = (at['t'] - ev[-1]) if ev else None
            out[f'{name}_{label}_count'] = len(ev)

    # Arrow kills: identify the arrow and its launch geometry.
    arrow_owner = at['players'][killer]['id'] if kind == 'deaths_opponent' else at['players'][dead]['id'] if kind == 'deaths_self' else None
    if arrow_owner is not None and p['cause_of_death'] in (0, 2):
        vx, vy = pos(before, dead)
        cand = [a for a in before['arrows'] if a['owner'] == arrow_owner]
        point_blank = False
        if not cand:
            cand = [a for a in at['arrows'] if a['owner'] == arrow_owner]
            point_blank = True
        if cand:
            k = min(cand, key=lambda a: dist((a['x'], a['y']), (vx, vy)))
            born = arrow_birth(frames, k['id'], idx - 1)
            bf = frames[born]
            ka = next((a for a in bf['arrows'] if a['id'] == k['id']), k)
            shooter = killer if kind == 'deaths_opponent' else dead
            launch_range = dist((ka['x'], ka['y']), pos(bf, dead))
            angle = math.degrees(math.atan2(k['vy'], k['vx']))
            out['arrow'] = dict(id=k['id'], flight_ticks=at['t'] - bf['t'], launch_range=launch_range, speed=math.hypot(k['vx'], k['vy']),
                                impact_angle_deg=angle, launch_dy=pos(bf, shooter)[1] - pos(bf, dead)[1],
                                launch_dx=abs(pos(bf, shooter)[0] - pos(bf, dead)[0]),
                                shooter_airborne=bool(slime_of(bf, shooter) and slime_of(bf, shooter)['state'] == STATE_FALL),
                                shooter_active=ABILITY_NAMES.get(active_ability(slime_of(bf, shooter))),
                                victim_active_at_launch=ABILITY_NAMES.get(active_ability(slime_of(bf, dead))),
                                victim_airborne_at_launch=bool(slime_of(bf, dead) and slime_of(bf, dead)['state'] == STATE_FALL),
                                victim_dash_remaining_at_launch=(ability_state(slime_of(bf, dead), DASH) or {}).get('remaining'),
                                victim_moved_during_flight=dist(pos(bf, dead), (vx, vy)), same_frame=point_blank,
                                live_enemy_arrows_at_launch=sum(1 for a in bf['arrows'] if a['owner'] == arrow_owner),
                                victim_dashed_during_flight=any(bf['t'] <= t <= at['t'] for t in event_ticks(frames, dead, 'dash_activations', born - 1, idx + 1)),
                                victim_jumped_during_flight=bool(button_presses(frames, dead, 1, born, idx + 1)))
            # The shooter's recent dash: dash->shoot combo (Matthew's signature attack).
            dashes = event_ticks(frames, shooter, 'dash_activations', WARMUP, born + 1)
            out['arrow']['shooter_dash_to_shot_ticks'] = (bf['t'] - dashes[-1]) if dashes else None
            vd = event_ticks(frames, dead, 'dash_activations', WARMUP, born + 1)
            out['arrow']['victim_dash_to_shot_ticks'] = (bf['t'] - vd[-1]) if vd else None

    # Squish: which platform, whose, and the escape window.
    if p['cause_of_death'] == 1:
        over = sorted(((platform_sdf(e, p['x'], p['y']), e) for e in at['platforms']), key=lambda t: t[0])
        if over:
            d, e = over[0]
            info = owners.get(e['id'])
            sq = dict(platform_id=e['id'], sdf=d, spawned=info is not None)
            if info:
                sq.update(owner=info['owner'], age_ticks=at['t'] - info['tick'], plant_distance=info['distance'])
                # morph start: first tick before the spawn where the owner had the platform ability active
                if info['owner'] is not None:
                    start = None
                    for f in reversed(frames[WARMUP:idx]):
                        s = slime_of(f, info['owner'])
                        st = ability_state(s, PLATFORM) if s else None
                        if st and st['active']:
                            start = f['t']
                        elif start is not None:
                            break
                    sq['morph_start_tick'] = start
                    sq['escape_window_ticks'] = (at['t'] - start) if start is not None else None
                    if start is not None:
                        f0 = frames[start]
                        sq['victim_distance_at_morph_start'] = dist(pos(f0, info['owner']), pos(f0, dead))
                        sq['victim_moved_in_window'] = dist(pos(f0, dead), (p['x'], p['y']))
                        sq['victim_dash_ready_at_morph_start'] = (ability_state(slime_of(f0, dead), DASH) or {}).get('ready')
                        sq['victim_active_at_morph_start'] = ABILITY_NAMES.get(active_ability(slime_of(f0, dead)))
            # ceiling above the victim: another platform within 4 units above (pinned)
            others = [platform_sdf(o, p['x'], p['y']) for _, o in over[1:]]
            sq['second_platform_sdf'] = min(others) if others else None
            # Crusher motion and the platform the victim stood on (PlayerCollision: squished = grounded on A, B overlaps the centre).
            prev = next((o for o in frames[max(WARMUP, idx - 30)]['platforms'] if o['id'] == e['id']), None)
            sq['crusher_speed'] = math.hypot(e['vx'], e['vy'])
            sq['crusher_moved_30'] = dist((e['x'], e['y']), (prev['x'], prev['y'])) if prev else None
            if sb is not None:
                gcand = sorted(((platform_sdf(o, sb['x'], sb['y'] - 0.5), o) for o in before['platforms'] if o['id'] != e['id']), key=lambda t: t[0])
                if gcand:
                    g = gcand[0][1]
                    ginfo = owners.get(g['id'])
                    gprev = next((o for o in frames[max(WARMUP, idx - 30)]['platforms'] if o['id'] == g['id']), None)
                    sq['ground_owner'] = ginfo['owner'] if ginfo else 'map'
                    sq['ground_sdf'] = gcand[0][0]
                    sq['ground_moved_30'] = dist((g['x'], g['y']), (gprev['x'], gprev['y'])) if gprev else None
            out['squish'] = sq

    # Falls: where did the victim leave the ground and what launched it.
    if kind == 'deaths_environment' and p['cause_of_death'] == 2:
        last_ground = None
        for j in range(idx - 1, WARMUP - 1, -1):
            s = slime_of(frames[j], dead)
            if s is not None and s['state'] != STATE_FALL:
                last_ground = j
                break
        fall = dict(fall_ticks=(at['t'] - frames[last_ground]['t']) if last_ground is not None else None)
        if last_ground is not None:
            lo = last_ground
            fall['from_x'], fall['from_y'] = pos(frames[lo], dead)
            fall['dash_uses'] = len(event_ticks(frames, dead, 'dash_activations', lo - 1, idx + 1))
            fall['platform_uses'] = len(event_ticks(frames, dead, 'platform_activations', lo - 1, idx + 1))
            fall['jump_presses'] = len(button_presses(frames, dead, 1, lo, idx + 1))
            fall['opponent_distance_at_launch'] = dist(pos(frames[lo], dead), pos(frames[lo], killer)) if slime_of(frames[lo], killer) else None
            fall['opponent_dashing_at_launch'] = active_ability(slime_of(frames[lo], killer)) == DASH
            fall['opponent_dashed_within_30'] = any(abs(t - frames[lo]['t']) <= 30 for t in event_ticks(frames, killer, 'dash_activations', WARMUP, idx + 1))
            fall['enemy_arrows_in_flight_at_launch'] = sum(1 for a in frames[lo]['arrows'] if a['owner'] == at['players'][killer]['id'])
            fall['dash_ready_at_launch'] = (ability_state(slime_of(frames[lo], dead), DASH) or {}).get('ready')
            fall['platform_ready_at_launch'] = (ability_state(slime_of(frames[lo], dead), PLATFORM) or {}).get('ready')
            # first action after leaving ground
            acts = []
            for t in event_ticks(frames, dead, 'dash_activations', lo - 1, idx + 1):
                acts.append((t, 'dash'))
            for t in event_ticks(frames, dead, 'platform_activations', lo - 1, idx + 1):
                acts.append((t, 'platform'))
            for t in button_presses(frames, dead, 1, lo, idx + 1):
                acts.append((t, 'jump'))
            acts.sort()
            fall['first_action'] = acts[0][1] if acts else None
            fall['first_action_delay'] = (acts[0][0] - frames[lo]['t']) if acts else None
            fall['actions'] = [a for _, a in acts]
            # was the victim shoved: velocity jump within 10 ticks of leaving ground while opponent within 3 units
            s0 = slime_of(frames[lo], dead)
            fall['launch_speed'] = math.hypot(s0['vx'], s0['vy']) if s0 else None
            fall['launch_vx'] = s0['vx'] if s0 else None
        out['fall'] = fall
    return out


def shot_facts(frames, player):
    """Every arrow release of `player`: shooter/opponent state at the release tick."""
    other = 1 - player
    out = []
    dashes = event_ticks(frames, player, 'dash_activations', WARMUP)
    for t in event_ticks(frames, player, 'arrows_fired', WARMUP):
        f = frames[t]
        s, o = slime_of(f, player), slime_of(f, other)
        if s is None:
            continue
        prior = [d for d in dashes if d <= t]
        out.append(dict(t=t, airborne=s['state'] == STATE_FALL, distance=dist(pos(f, player), pos(f, other)) if o else None,
                        dy=(pos(f, player)[1] - pos(f, other)[1]) if o else None, dash_age=(t - prior[-1]) if prior else None,
                        opponent_active=ABILITY_NAMES.get(active_ability(o)) if o else None, opponent_airborne=bool(o and o['state'] == STATE_FALL),
                        opponent_dash_ready=(ability_state(o, DASH) or {}).get('ready') if o else None))
    return out


def bow_draw_facts(frames, player):
    """Every bow morph of `player`: commitment window (morph start -> release or death) and the opponent's
    range / dash state when the morph was committed. `died` = the drawer died before releasing."""
    other = 1 - player
    out = []
    active = False
    start = None
    releases = event_ticks(frames, player, 'arrows_fired', WARMUP)
    last = len(frames) - 1
    for i in range(WARMUP, len(frames)):
        f = frames[i]
        s = slime_of(f, player)
        st = ability_state(s, BOW) if s else None
        now = bool(st and st['active'])
        alive = bool(f['players'][player]['flags'] & 1)
        if now and not active:
            start = i
        # the active bit stays set after the release until the morph ends, so a draw that is still
        # "active" when the round ends (the winner's last shot) is flushed at the last frame
        if active and (not now or not alive or i == last):
            f0 = frames[start]
            o0 = slime_of(f0, other)
            od = ability_state(o0, DASH) if o0 else None
            rel = [t for t in releases if f0['t'] <= t <= f['t']]
            # did the opponent dash inside the window, and from how far
            odash = [t for t in event_ticks(frames, other, 'dash_activations', start - 1, i + 1)]
            odash_range = dist(pos(frames[odash[0]], player), pos(frames[odash[0]], other)) if odash and slime_of(frames[odash[0]], other) else None
            out.append(dict(start=f0['t'], end=f['t'], ticks=f['t'] - f0['t'], released=bool(rel), release_ticks=(rel[0] - f0['t']) if rel else None,
                            died=not alive, distance=dist(pos(f0, player), pos(f0, other)) if o0 else None,
                            dy=(pos(f0, player)[1] - pos(f0, other)[1]) if o0 else None,
                            opponent_dash_ready=od['ready'] if od else None, opponent_dash_remaining=od['remaining'] if od else None,
                            opponent_active=ABILITY_NAMES.get(active_ability(o0)) if o0 else None,
                            opponent_airborne=bool(o0 and o0['state'] == STATE_FALL), opponent_dashed=bool(odash),
                            opponent_dash_range=odash_range, opponent_dash_delay=(odash[0] - f0['t']) if odash else None,
                            opponent_arrows_in_flight=sum(1 for a in f0['arrows'] if a['owner'] == f0['players'][other]['id']),
                            airborne=bool(slime_of(f0, player) and slime_of(f0, player)['state'] == STATE_FALL)))
            start = None
        active = now and alive
    return out


def platform_use_facts(frames, player):
    other = 1 - player
    out = []
    for t in event_ticks(frames, player, 'platform_activations', WARMUP):
        f = frames[t]
        o = slime_of(f, other)
        out.append(dict(t=t, distance=dist(pos(f, player), pos(f, other)) if o else None, dy=(pos(f, player)[1] - pos(f, other)[1]) if o else None))
    return out


def round_facts(game):
    row, frames, check = game['row'], game['frames'], game['check']
    tel = frames[-1]['telemetry']
    owners = platform_owners(frames)
    out = dict(name=game['name'], agent=agent_name(row), tier=tier(row), stratum=stratum(row), map=row['map'], session=row['session'], time=row['time'],
               outcome=row['outcome'], ticks=row['ticks'], seed=row['seed'], verified=bool(check['ticks_match'] and check['outcome_match'] and check['causes_match']),
               human_abilities=[ABILITY_NAMES[a] for a in row['human_abilities'][0]],
               human_arrows=tel[0]['arrows_fired'], agent_arrows=tel[1]['arrows_fired'], human_dashes=tel[0]['dash_activations'], agent_dashes=tel[1]['dash_activations'],
               human_platforms=tel[0]['platform_activations'], agent_platforms=tel[1]['platform_activations'],
               spawned_platforms=Counter(str(v['owner']) for v in owners.values()),
               shots=[shot_facts(frames, 0), shot_facts(frames, 1)], platform_uses=[platform_use_facts(frames, 0), platform_use_facts(frames, 1)],
               bow_draws=[bow_draw_facts(frames, 0), bow_draw_facts(frames, 1)],
               opponent=row.get('opponent'), agent_name=row.get('agent_name'), timeout=row.get('timeout', False))
    # mean separation over the round (both alive)
    ds = [dist(pos(f, 0), pos(f, 1)) for f in frames[WARMUP::4] if slime_of(f, 0) and slime_of(f, 1)]
    out['mean_distance'] = (sum(ds) / len(ds)) if ds else None
    dead = [i for i in (0, 1) if not frames[-1]['players'][i]['flags'] & 1]
    out['dead'] = dead
    if dead:
        out['death'] = death_facts(frames, dead[0] if len(dead) == 1 else (1 if row['outcome'] == 'a' else 0), owners)
    return out


NOVEL = {'Grenade', 'Drill', 'Magnet gun', 'Mine', 'Poke ball', 'Blink gun', 'Beam'}
POINT_BLANK = 5.0      # units; the bow's own hurtbox is ~1.3 wide, a slime ~1.7
DASH_SHOT = 60         # ticks; dash -> release inside one second is one motion
LONG_FLIGHT = 15       # ticks; >= 4 decisions of a visible arrow


def classify(d, human_abilities):
    """Primary pattern + tags for one death record. Names read as <cause>/<setup>."""
    tags = []
    victim = 'agent' if d['dead'] == 1 else 'human'
    cause, kind = d['cause'], d['killer_kind']
    a = d.get('arrow')
    if cause == 'other' and kind == 'opponent':
        if a is None:
            # no arrow found: the human's non-standard weapon (grenade, drill, mine, poke ball...) or a body kill
            novel = [n for n in human_abilities if n in NOVEL] if victim == 'agent' else []
            return ('novel-weapon/' + '+'.join(novel)) if novel else 'other-opponent-kill', tags
        if a['shooter_dash_to_shot_ticks'] is not None and a['shooter_dash_to_shot_ticks'] <= DASH_SHOT:
            tags.append('dash-in')
        if a['shooter_airborne']:
            tags.append('shooter-airborne')
        if a['launch_dy'] > 4:
            tags.append('from-above')
        elif a['launch_dy'] < -4:
            tags.append('from-below')
        if a['victim_airborne_at_launch']:
            tags.append('victim-airborne')
        if a['victim_dash_remaining_at_launch'] is not None and a['victim_dash_remaining_at_launch'] > 0:
            tags.append('victim-dash-on-cooldown')
        if a['victim_dash_to_shot_ticks'] is not None and a['victim_dash_to_shot_ticks'] <= 90:
            tags.append('victim-just-dashed')
        if a['victim_jumped_during_flight']:
            tags.append('victim-jumped-into-it')
        va = a['victim_active_at_launch'] or d['victim_active_ability']
        if va == 'Bow':
            return 'arrow/hit-while-drawing-bow', tags
        if va == 'Platform':
            return 'arrow/hit-while-platform-morph', tags
        if va == 'Dash' or (a['victim_dash_to_shot_ticks'] is not None and a['victim_dash_to_shot_ticks'] <= 24):
            return 'arrow/hit-mid-dash', tags
        if a['launch_range'] < POINT_BLANK:
            return ('arrow/point-blank-after-dash' if 'dash-in' in tags else 'arrow/point-blank'), tags
        if 'dash-in' in tags:
            return 'arrow/dash-in-shot', tags
        if a['flight_ticks'] >= LONG_FLIGHT and a['victim_moved_during_flight'] < 1.5:
            return 'arrow/long-flight-not-dodged', tags
        if a['flight_ticks'] >= LONG_FLIGHT:
            return 'arrow/long-flight-dodge-failed', tags
        return 'arrow/short-range', tags
    if cause == 'beamed':
        return 'novel-weapon/Beam', tags
    if cause == 'accident' and kind == 'self':
        if a is None:
            return 'self/own-weapon', tags
        if a['flight_ticks'] >= 15 and a['launch_range'] < 1.0 and a['impact_angle_deg'] < -90:
            return 'self/own-arrow-mortar-returned', tags
        return 'self/own-arrow', tags
    if cause == 'squished':
        sq = d.get('squish') or {}
        owner = sq.get('owner') if sq.get('spawned') else 'map'
        me = d['dead']
        foe = 1 - me
        ground = sq.get('ground_owner')
        moving = (sq.get('crusher_moved_30') or 0) > 1.0 or (sq.get('crusher_speed') or 0) > 1.5
        if ground == me:
            tags.append('standing-on-own-platform')
        if d['killer_active_ability'] == 'Platform':
            tags.append('foe-in-platform-morph')
        if owner == foe:
            if (sq.get('escape_window_ticks') or 999) <= 31:
                return 'squish/foe-platform-dropped-on-victim', tags
            if moving:
                return 'squish/foe-platform-rammed-victim', tags
            return 'squish/victim-moved-into-foe-platform', tags
        if owner == me:
            if d['victim_active_ability'] == 'Platform':
                return ('squish/own-platform-exit-pinch-contested' if 'foe-in-platform-morph' in tags else 'squish/own-platform-exit-pinch'), tags
            return 'squish/own-platform-pushed-into-victim', tags
        # map platform is the crusher
        if moving:
            return 'squish/moving-map-platform', tags
        if ground == foe and (sq.get('ground_moved_30') or 0) > 1.0:
            return 'squish/carried-into-map-on-foe-platform', tags
        if ground == me:
            return 'squish/own-platform-drifted-into-map', tags
        return 'squish/map-platform-other', tags
    if cause == 'accident' and kind == 'environment':
        fl = d.get('fall') or {}
        if not fl or fl.get('fall_ticks') is None:
            return 'fall/unknown', tags
        if fl['fall_ticks'] <= 2:
            return 'fall/ground-reached-water', tags
        if (fl.get('opponent_distance_at_launch') or 99) < 4 and fl.get('opponent_dashed_within_30'):
            return 'fall/shoved-off', tags
        if fl.get('enemy_arrows_in_flight_at_launch'):
            tags.append('arrow-in-flight')
        if fl.get('dash_uses'):
            tags.append('dashed-in-air')
        if fl.get('platform_uses'):
            tags.append('platform-in-air')
        if fl.get('first_action') == 'jump' and (fl.get('first_action_delay') or 0) <= 2:
            return ('fall/jumped-off-then-platform-failed' if fl.get('platform_uses') else 'fall/jumped-off-edge'), tags
        if fl.get('first_action') == 'dash':
            return 'fall/dashed-off-edge', tags
        return 'fall/walked-or-knocked-off', tags
    if cause == 'other' and kind == 'environment':
        return 'environment/level-hazard', tags
    return f'{cause}/{kind}', tags


def facts(args):
    paths = sorted(glob.glob(str(args.extracted / 'games' / '*.json.gz')))
    out = []
    for i, p in enumerate(paths):
        game = load_game(p)
        try:
            out.append(round_facts(game))
        except Exception as error:
            out.append(dict(name=game['name'], error=repr(error)))
            raise
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(paths)}', flush=True)
    args.out.write_text(json.dumps(out, indent=0, default=str))
    print(f'wrote {args.out} ({len(out)} rounds)')


STRATA = ('A', 'B', 'C', 'D')


def md_table(header, rows):
    out = ['| ' + ' | '.join(str(h) for h in header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out) + '\n'


def pct(n, d):
    return f'{n} ({100 * n / d:.0f}%)' if d else '0'


def deaths(rounds, victim):
    """(round, death, pattern, tags) for every round whose (single) death is of `victim` (0 human, 1 agent)."""
    out = []
    for r in rounds:
        d = r.get('death')
        if d and d['dead'] == victim:
            p, tags = classify(d, r['human_abilities'])
            out.append((r, d, p, tags))
    return out


def pattern_table(rounds, victim, groups=STRATA, group_of=lambda r: r['stratum'], family_only=False):
    ds = deaths(rounds, victim)
    key = (lambda p: p.split('/')[0]) if family_only else (lambda p: p)
    counts = Counter((key(p), group_of(r)) for r, d, p, t in ds)
    totals = Counter(key(p) for r, d, p, t in ds)
    per_group = Counter(group_of(r) for r, d, p, t in ds)
    rows = []
    for p, n in totals.most_common():
        rows.append([p, pct(n, len(ds))] + [counts.get((p, g), 0) for g in groups])
    rows.append(['**all**', len(ds)] + [per_group.get(g, 0) for g in groups])
    return md_table(['pattern', 'all'] + list(groups), rows)


def shots_of(rounds, player):
    return [s for r in rounds for s in r['shots'][player]]


def kill_shots(rounds, victim):
    return [d['arrow'] for r, d, p, t in deaths(rounds, victim) if d.get('arrow') and d['killer_kind'] == 'opponent']


def shot_style(shots, kills):
    def frac(xs, pred):
        xs = list(xs)
        return pct(sum(1 for x in xs if pred(x)), len(xs))
    return dict(shots=len(shots), kills=len(kills),
                dash_in_all=frac(shots, lambda s: s['dash_age'] is not None and s['dash_age'] <= DASH_SHOT),
                airborne_all=frac(shots, lambda s: s['airborne']),
                close_all=frac(shots, lambda s: s['distance'] is not None and s['distance'] < 10),
                dash_in_kills=frac(kills, lambda a: a['shooter_dash_to_shot_ticks'] is not None and a['shooter_dash_to_shot_ticks'] <= DASH_SHOT),
                unreactable_kills=frac(kills, lambda a: a['shooter_dash_to_shot_ticks'] is not None and a['shooter_dash_to_shot_ticks'] <= 13),
                airborne_kills=frac(kills, lambda a: a['shooter_airborne']),
                pb_kills=frac(kills, lambda a: a['launch_range'] < POINT_BLANK))


def tables(args):
    F = json.load(open(args.facts))
    sim = json.load(open(args.sim_facts)) if args.sim_facts else []
    out = []
    P = out.append
    unverified = [r for r in F if not r['verified']]
    if unverified and not args.all:
        # a replay that does not reproduce the logged end tick / outcome / causes is a different game: leave it out
        P(f"_{len(unverified)} of {len(F)} rounds do not reproduce the logged outcome and are excluded: {', '.join(r['name'] for r in unverified)}_\n")
        F = [r for r in F if r['verified']]
    P('## Rounds and outcomes by stratum\n')
    rows = []
    for g in STRATA:
        rs = [r for r in F if r['stratum'] == g]
        hw = sum(r['outcome'] == 'a' for r in rs)
        strong = [r for r in rs if r['tier'] == 'strong-checkpoint']
        rows.append([g, len(rs), hw, len(rs) - hw, f'{100 * hw / len(rs):.0f}%' if rs else '-', len(strong), sum(r['outcome'] == 'a' for r in strong),
                     len({r['map'] for r in rs}), sum(r['verified'] for r in rs)])
    P(md_table(['stratum', 'rounds', 'human wins', 'agent wins', 'human win %', 'vs strong', 'human wins vs strong', 'maps', 'replay-verified'], rows))
    P('\n## Per-tier outcomes by stratum\n')
    rows = []
    for t in list(TIERS) + ['other-checkpoint', 'bot']:
        rs = [r for r in F if r['tier'] == t]
        if not rs:
            continue
        rows.append([t, len(rs), pct(sum(r['outcome'] == 'a' for r in rs), len(rs))] + [f"{sum(r['outcome'] == 'a' for r in rs if r['stratum'] == g)}/{sum(r['stratum'] == g for r in rs)}" for g in STRATA])
    P(md_table(['tier', 'rounds', 'human wins'] + [f'{g} (W/n)' for g in STRATA], rows))
    P('\n## Agent deaths (human wins) by pattern and stratum\n')
    P(pattern_table(F, 1))
    P('\n## Agent deaths by family and stratum\n')
    P(pattern_table(F, 1, family_only=True))
    P('\n## Human deaths (agent wins) by pattern and stratum\n')
    P(pattern_table(F, 0))
    P('\n## Agent deaths by pattern and agent tier\n')
    tiers = ('strong-checkpoint', 'mid-checkpoint', 'weak-checkpoint', 'other-checkpoint', 'bot')
    P(pattern_table(F, 1, groups=tiers, group_of=lambda r: r['tier']))
    P('\n## Agent deaths per map (stratum B, maps with >= 3 agent deaths)\n')
    B = [r for r in F if r['stratum'] == 'B']
    per_map = Counter(r['map'] for r, d, p, t in deaths(B, 1))
    rows = []
    for m, n in per_map.most_common():
        if n < 3:
            break
        fam = Counter(p.split('/')[0] for r, d, p, t in deaths(B, 1) if r['map'] == m)
        rows.append([m, sum(r['map'] == m for r in B), n, ', '.join(f'{k} {v}' for k, v in fam.most_common())])
    P(md_table(['map', 'rounds', 'agent deaths', 'families'], rows))
    P('\n## Tags on agent arrow deaths by stratum\n')
    ads = [(r, d, p, t) for r, d, p, t in deaths(F, 1) if p.startswith('arrow/')]
    tagc = Counter((tag, r['stratum']) for r, d, p, t in ads for tag in t)
    tag_all = Counter(tag for r, d, p, t in ads for tag in t)
    P(md_table(['tag', 'all arrow deaths'] + list(STRATA), [[k, pct(v, len(ads))] + [tagc.get((k, g), 0) for g in STRATA] for k, v in tag_all.most_common()]))
    P('\n## Pre-death window (120 ticks) of the AGENT: rounds the agent lost vs won\n')
    rows = []
    fields = ('distance_mean', 'distance_min', 'airborne_fraction', 'dash_ready_end', 'platform_ready_end', 'bow_ready_end', 'arrows_fired_window', 'opponent_arrows_window',
              'ticks_since_last_shot', 'ticks_since_last_dash', 'height_advantage_end', 'speed_mean')
    for g in ('all',) + STRATA:
        rs = [r for r in F if g == 'all' or r['stratum'] == g]
        lost = [d['victim_window'] for r, d, p, t in deaths(rs, 1)]
        won = [d['killer_window'] for r, d, p, t in deaths(rs, 0)]
        for name, ws in (('agent lost', lost), ('agent won', won)):
            row = [g, name, len(ws)]
            for f in fields:
                xs = [w[f] for w in ws if w.get(f) is not None]
                if not xs:
                    row.append('-')
                elif isinstance(xs[0], bool):
                    row.append(f'{100 * sum(xs) / len(xs):.0f}%')
                else:
                    row.append(f'{statistics.median(xs):.1f}')
            rows.append(row)
    P(md_table(['stratum', 'outcome', 'n'] + [f.replace('_', ' ') for f in fields], rows))
    P('\n## Shot style: every arrow release and every kill shot\n')
    rows = []
    def style_row(label, shots, kills):
        s = shot_style(shots, kills)
        rows.append([label, s['shots'], s['dash_in_all'], s['airborne_all'], s['close_all'], s['kills'], s['dash_in_kills'], s['unreactable_kills'], s['airborne_kills'], s['pb_kills']])
    for g in ('all',) + STRATA:
        rs = [r for r in F if g == 'all' or r['stratum'] == g]
        std = [r for r in rs if 'Bow' in r['human_abilities']]
        style_row(f'human ({g})', shots_of(std, 0), kill_shots(std, 1))
        style_row(f'agent ({g})', shots_of(rs, 1), kill_shots(rs, 0))
    for opp in sorted({r['opponent'] for r in sim}) if sim else []:
        rs = [r for r in sim if r['opponent'] == opp]
        style_row(f'unity: {opp} (seat 0)', shots_of(rs, 0), kill_shots(rs, 1))
    for ag in sorted({r['agent_name'] for r in sim}) if sim else []:
        rs = [r for r in sim if r['agent_name'] == ag]
        style_row(f'unity: {ag} (seat 1)', shots_of(rs, 1), kill_shots(rs, 0))
    P(md_table(['shooter', 'shots', 'dash<=60 before release', 'airborne', 'range<10', 'kill shots', 'dash<=60', 'dash<=13 (unreactable)', 'airborne', 'point-blank'], rows))
    P('\n## Bow-draw commitment window\n')
    rows = []
    def draw_rows(label, draws):
        rel = [d for d in draws if d['released']]
        died = [d for d in draws if d['died']]
        hist = Counter(min(d['release_ticks'] // 10 * 10, 100) for d in rel)
        rows.append([label, len(draws), len(rel), f"{statistics.median([d['release_ticks'] for d in rel]):.0f}" if rel else '-',
                     ' '.join(f'{k}:{hist[k]}' for k in sorted(hist)), len(died),
                     f"{statistics.median([d['ticks'] for d in died]):.0f}" if died else '-',
                     f"{statistics.median([d['distance'] for d in died if d['distance'] is not None]):.1f}" if died else '-',
                     pct(sum(bool(d['opponent_dash_ready']) for d in died), len(died)), pct(sum(d['opponent_dashed'] for d in died), len(died)),
                     pct(sum(d['opponent_airborne'] for d in died), len(died)), pct(sum(d['opponent_active'] == 'Bow' for d in died), len(died))])
    for g in ('all',) + STRATA:
        rs = [r for r in F if g == 'all' or r['stratum'] == g]
        draw_rows(f'agent ({g})', [d for r in rs for d in r['bow_draws'][1]])
        draw_rows(f'human ({g})', [d for r in rs for d in r['bow_draws'][0] if 'Bow' in r['human_abilities']])
    for opp in sorted({r['opponent'] for r in sim}) if sim else []:
        rs = [r for r in sim if r['opponent'] == opp]
        draw_rows(f'unity: agents vs {opp}', [d for r in rs for d in r['bow_draws'][1]])
    P(md_table(['drawer', 'draws', 'released', 'median release tick', 'release histogram (10-tick bins)', 'died drawing', 'median window', 'opp. distance at commit',
                'opp. dash ready at commit', 'opp. dashed in window', 'opp. airborne at commit', 'opp. drawing at commit'], rows))
    P('\n## Death rate of a bow draw by opponent distance at commit\n')
    rows = []
    for label, draws in (('agent', [d for r in F for d in r['bow_draws'][1]]), ('human', [d for r in F for d in r['bow_draws'][0] if 'Bow' in r['human_abilities']])):
        for lo, hi in ((0, 6), (6, 10), (10, 15), (15, 25), (25, 99)):
            sub = [d for d in draws if d['distance'] is not None and lo <= d['distance'] < hi]
            dr = [d for d in sub if d['opponent_dash_ready']]
            rows.append([label, f'{lo}-{hi}', len(sub), pct(sum(d['died'] for d in sub), len(sub)), pct(sum(d['died'] for d in dr), len(dr)), pct(sum(d['died'] for d in sub if not d['opponent_dash_ready']), len(sub) - len(dr))])
    P(md_table(['drawer', 'opp. distance', 'draws', 'died', 'died | opp. dash ready', 'died | opp. dash not ready'], rows))
    if sim:
        P('\n## Unity agent-vs-agent rounds (same build, same taxonomy): agent (seat 1) deaths by pattern and opponent\n')
        opps = sorted({r['opponent'] for r in sim})
        P(pattern_table(sim, 1, groups=opps, group_of=lambda r: r['opponent']))
        P('\n### seat-0 opponent deaths by pattern (how the bots / pool members die to the agent)\n')
        P(pattern_table(sim, 0, groups=opps, group_of=lambda r: r['opponent']))
        P('\n### agent deaths by family: training maps 0-2 vs unseen maps, per agent\n')
        rows = []
        for ag in sorted({r['agent_name'] for r in sim}):
            for label, pred in (('maps 0-2', lambda r: r['map'] in (0, 1, 2)), ('maps 3-4', lambda r: r['map'] in (3, 4)), ('unseen', lambda r: r['map'] > 4)):
                rs = [r for r in sim if r['agent_name'] == ag and pred(r)]
                ds = deaths(rs, 1)
                fam = Counter(p.split('/')[0] for r, d, p, t in ds)
                pin = sum(1 for r, d, p, t in ds if p.startswith('squish/own-platform'))
                rows.append([ag, label, len(rs), pct(sum(r['outcome'] == 'a' for r in rs), len(rs)), sum(r['outcome'] == 'draw' for r in rs), len(ds)] + [pct(fam.get(k, 0), len(ds)) for k in ('arrow', 'squish', 'fall', 'self')] + [pct(pin, len(ds))])
        P(md_table(['agent', 'maps', 'rounds', 'agent lost', 'draws', 'agent deaths', 'arrow', 'squish', 'fall', 'self', 'own-platform squish'], rows))
    text = '\n'.join(out)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('facts')
    f.add_argument('--extracted', type=Path, required=True)
    f.add_argument('--out', type=Path, required=True)
    f.set_defaults(func=facts)
    t = sub.add_parser('tables', help='markdown tables for the report')
    t.add_argument('--facts', type=Path, required=True)
    t.add_argument('--sim-facts', type=Path, default=None, help='facts.json of agent_games.py rounds')
    t.add_argument('--out', type=Path, default=None)
    t.add_argument('--all', action='store_true', help='keep rounds whose replay does not reproduce the logged outcome')
    t.set_defaults(func=tables)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
