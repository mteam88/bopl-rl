"""Loss taxonomy for the scripted bots: play bots against checkpoints (and each other),
save every game (native .rep, physics-state visual .json.gz, bot decision log), then
classify how each game ended from the saved frames.

    python loss_taxonomy.py play --out DIR --checkpoint name=path ... [--bots archer,pursuer,evasive,mixed]
                                 [--maps 0,1,2] [--seed-pairs 7] [--bot-vs-bot-pairs 4] [--workers 6]
    python loss_taxonomy.py classify --out DIR            # writes DIR/taxonomy.json (one row per game)

`play` wires a checkpoint to a player exactly like evaluate.py (native CPU PufferNet,
masked multinomial sampling with a per-game torch generator, the checkpoint's
observation delay, 7200-tick limit); scripted bots decide inside the C environment.
Set BOPL_RESEARCH_LIBRARY to a research library built with BOPL_BOT_DEBUG output for
per-decision bot internals (eta/dodge/veto); the log is captured per game.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from evaluation_protocol import SCRIPTED_KINDS as BOTS  # noqa: E402  kind name -> id (v1 and v2 kinds)
from map_catalogue import IDENTITY, normalize_record, parse_map_ids  # noqa: E402
HARD = 2
# Death geometry, Assets/Scripts/Assembly-CSharp/SceneBounds.cs + DestroyIfOutsideSceneBounds.UpdateSim
WATER_Y = -11.3
BLAST_X = 131.25
BLAST_Y = 72.5
DODGE_WINDOW_HARD = 26  # bopl_bots.h bot_dodge: hard level window (ticks)


# ---------------------------------------------------------------- play ----
def _schedule(checkpoints, bots, maps, seed_pairs, bvb_pairs):
    tasks = []
    for bot in bots:
        for name in checkpoints:
            for map_id in maps:
                for pair in range(seed_pairs):
                    seed = 50001 + 100000 * map_id + pair
                    for side in (0, 1):  # side = policy player index; bot is 1-side
                        tasks.append(dict(kind='policy', bot=bot, opponent=name, map=map_id, seed=seed,
                                          side=side, name=f'{bot}-vs-{name}-m{map_id}-s{pair}-p{side}'))
    bot_list = list(bots)
    for i, a in enumerate(bot_list):
        for b in bot_list[i + 1:]:
            for map_id in maps:
                for pair in range(bvb_pairs):
                    seed = 70001 + 100000 * map_id + pair
                    for side in (0, 1):  # side = player index of bot `a`
                        tasks.append(dict(kind='bots', bot=a, opponent=b, map=map_id, seed=seed,
                                          side=side, name=f'{a}-vs-{b}-m{map_id}-s{pair}-p{side}'))
    return tasks


def _frame(env):
    """evaluate.py's physics frame plus each slime's ability_flags (needed to tell morphs apart)."""
    frame = env.frame()
    s = env.state
    for i, slime in enumerate(frame['slimes']):
        slime['ability_flags'] = list(s.slimes[i].ability_flags)
    return frame


def _worker(args):
    out, checkpoints, tasks, worker_index = args
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(HERE))
    os.environ.setdefault('BOPL_RL_INPROC_SPARE', '0')
    os.environ.setdefault('BOPL_RL_WORLDS_PER_RUNTIME', str(1 << 20))
    os.environ['BOPL_BOT_DEBUG'] = '1'
    from environment import Environment, configure
    from native_policy import load
    from native_policy import decide
    import layout
    configure(out / 'workers', ROOT / 'native/target/release/libbopl_sim.so')
    models = {name: load(Path(path)) for name, path in checkpoints.items()}
    games = out / 'games'
    games.mkdir(exist_ok=True)
    results = []
    any_meta = next(iter(models.values()))[1]
    delay = any_meta['config'].get('observation_delay_ticks', 9)
    limit = any_meta['config']['episode_ticks']
    for task in tasks:
        policy = meta = None
        if task['kind'] == 'policy':
            policy, meta = models[task['opponent']]
        # Capture the C bot's stderr debug stream per game.
        log_path = games / (task['name'] + '.bot.log')
        saved = os.dup(2)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(fd, 2)
        os.close(fd)
        try:
            with Environment(seed=90001 + worker_index, limit=limit, delay=delay) as env:
                env.reset(task['seed'], task['map'])
                env.bot(0, 0)
                env.bot(1, 0)
                if task['kind'] == 'policy':
                    env.bot(1 - task['side'], BOTS[task['bot']], HARD)
                else:
                    env.bot(task['side'], BOTS[task['bot']], HARD)
                    env.bot(1 - task['side'], BOTS[task['opponent']], HARD)
                generator = torch.Generator().manual_seed(task['seed'] * 7 + task['side'])
                state = policy.initial(2, 'cpu') if policy is not None else None
                frames = [_frame(env)]
                while not env.ended:
                    if policy is not None:
                        observation = torch.from_numpy(env.obs.copy())
                        mask = env.mask.copy()
                        with torch.no_grad():
                            logits, values, actions, state = decide(policy, observation, mask, state, generator)
                        actions = actions.numpy()
                    else:
                        actions = np.zeros((2, layout.NUM_ATNS), dtype=np.float32)
                    frames[-1]['actions'] = actions.tolist()
                    env.step(actions)
                    frames.append(_frame(env))
                final = frames[-1]
                env.replay(games / (task['name'] + '.rep'))
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        timeout = bool(final['flags'] & 2)
        players = final['players']
        # Two winners in a 1v1 is the game flagging both sides of a round it
        # left undecided; nobody won it, and the reward pays nobody
        # (bopl_undecided in native/rl/puffer/bopl_puffer.h).
        decided = not timeout and sum(bool(p['flags'] & 2) for p in players) == 1
        winner = next((i for i, p in enumerate(players) if p['flags'] & 2), None) if decided else None
        bot_index = (1 - task['side']) if task['kind'] == 'policy' else task['side']
        result = dict(task, timeout=timeout, ticks=final['t'], winner=winner, bot_index=bot_index,
                      bot_won=(winner == bot_index), final_players=players, telemetry=final['telemetry'],
                      visual=task['name'] + '.json.gz', replay=task['name'] + '.rep', bot_log=task['name'] + '.bot.log.gz')
        visual = dict(schema=1, step=meta['step'] if meta else 0, task=task, result={k: result[k] for k in ('timeout', 'ticks', 'winner', 'bot_index', 'bot_won')},
                      frames=frames, rendering='physics state replay', tick_rate=60, decision_rate=15,
                      actions_timing='observation-before-action')
        with gzip.open(games / (task['name'] + '.json.gz'), 'wt', compresslevel=4) as handle:
            json.dump(visual, handle, separators=(',', ':'))
        with open(log_path, 'rb') as src, gzip.open(games / (task['name'] + '.bot.log.gz'), 'wb', compresslevel=4) as dst:
            dst.write(src.read())
        log_path.unlink()
        results.append(result)
        print(json.dumps({k: result[k] for k in ('name', 'timeout', 'ticks', 'winner', 'bot_won')}), flush=True)
    return results


def play(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition('=')
        checkpoints[name] = str(Path(path).resolve())
    maps=parse_map_ids(args.maps)
    tasks = _schedule(checkpoints, args.bots.split(','), maps,
                      args.seed_pairs, args.bot_vs_bot_pairs)
    done = set()
    manifest = out / 'games.jsonl'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                done.add(json.loads(line)['name'])
            except ValueError:
                pass
    todo = [t for t in tasks if t['name'] not in done]
    print(f'{len(tasks)} games scheduled, {len(todo)} to play, {args.workers} workers', flush=True)
    (out / 'schedule.json').write_text(json.dumps(dict(map_catalogue=IDENTITY,
                                                       checkpoints=checkpoints,tasks=tasks),indent=1))
    # Interleave so long (timeout) games spread over workers.
    buckets = [todo[i::args.workers] for i in range(args.workers)]
    started = time.monotonic()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool, manifest.open('a') as handle:
        for results in pool.imap_unordered(_worker, [(out, checkpoints, b, i) for i, b in enumerate(buckets) if b]):
            for result in results:
                handle.write(json.dumps(result) + '\n')
            handle.flush()
    print(f'done in {time.monotonic() - started:.0f}s', flush=True)


# ------------------------------------------------------------ classify ----
LOG_RE = re.compile(r'\[bot (\d)\] d(\d+) t(\d+) me\(([-\d.]+),([-\d.]+)\) v\(([-\d.]+),([-\d.]+)\) g(\d) m(\d) foe\(([-\d.]+),([-\d.]+)\) mode(\d) bow(\d) dash(\d)(?: plat(\d))? (LATCHED held(\w+) left(\d+) aim\s*(\d+) eta(\d+|inf|1000000000) arr(\d+)|aim\s*(\d+) btn(\w+) hold(\d) spot(\d)\(([-\d.]+),([-\d.]+)\) land\[(\d)(\d)(\d)\] arr(\d+) eta(\d+|inf|1000000000) dodge(-?\d) veto(\d) still(\d+))')


def parse_log(path):
    """Per-decision bot internals from the captured BOPL_BOT_DEBUG stream (see loss_taxonomy README section)."""
    rows = []
    with gzip.open(path, 'rt', errors='replace') as handle:
        for line in handle:
            m = LOG_RE.match(line)
            if not m:
                continue
            g = m.groups()
            row = dict(player=int(g[0]), decision=int(g[1]), t=int(g[2]), x=float(g[3]), y=float(g[4]),
                       vx=float(g[5]), vy=float(g[6]), grounded=int(g[7]), morphing=int(g[8]),
                       fx=float(g[9]), fy=float(g[10]), mode=int(g[11]), bow=int(g[12]), dash=int(g[13]),
                       plat=int(g[14]) if g[14] is not None else None)
            if g[15].startswith('LATCHED'):
                row.update(latched=True, held=int(g[16], 16), left=int(g[17]), aim=int(g[18]),
                           eta=float(g[19]), arrows=int(g[20]), buttons=0, dodge=None, veto=0, spot=None)
            else:
                row.update(latched=False, aim=int(g[21]), buttons=int(g[22], 16), hold=int(g[23]),
                           spot=int(g[24]), spot_x=float(g[25]), spot_y=float(g[26]),
                           land=(int(g[27]), int(g[28]), int(g[29])), arrows=int(g[30]), eta=float(g[31]),
                           dodge=int(g[32]), veto=int(g[33]), still=int(g[34]))
            rows.append(row)
    return rows


def arrows_in(frame):
    return [e for e in frame['entities'] if e['kind'] == 0 and e['layer'] == 9]


def dash_ready(slime):
    """bopl_bots.h bot_view: an ability is ready when its timer exceeds its cooldown (slot 0 = dash)."""
    return slime['timers'][0] > slime['cooldowns'][0]


def event_ticks(frames, player, field):
    """Ticks at which a player's cumulative telemetry counter incremented (decision resolution, 4 ticks)."""
    ticks = []
    last = frames[0]['telemetry'][player][field]
    for f in frames[1:]:
        value = f['telemetry'][player][field]
        if value > last:
            ticks.extend([f['t']] * (value - last))
        last = value
    return ticks


def combos(frames, player, window=30):
    """dash->arrow and arrow->dash pairs within `window` ticks (Matthew: dash+shoot is THE attack pattern)."""
    dashes = event_ticks(frames, player, 'dash_activations')
    arrows = event_ticks(frames, player, 'arrows_fired')
    dash_then_arrow = sum(1 for d in dashes if any(0 <= a - d <= window for a in arrows))
    arrow_then_dash = sum(1 for a in arrows if any(0 < d - a <= window for d in dashes))
    return dict(dashes=len(dashes), arrows=len(arrows), dash_then_arrow=dash_then_arrow, arrow_then_dash=arrow_then_dash)


def cooldown_windows(frames, shooter, target):
    """How the shooter's arrow releases distribute over the target's dash-cooldown state.

    Returns ticks with the target's dash down/up (both alive) and the shooter's arrows released in each."""
    down = up = 0
    arrows_down = arrows_up = 0
    last_arrows = frames[0]['telemetry'][shooter]['arrows_fired']
    for a, b in zip(frames, frames[1:]):
        if not (a['players'][0]['flags'] & 1 and a['players'][1]['flags'] & 1):
            break
        if len(a['slimes']) <= target:
            continue
        ready = dash_ready(a['slimes'][target])
        dt = b['t'] - a['t']
        fired = b['telemetry'][shooter]['arrows_fired'] - last_arrows
        last_arrows = b['telemetry'][shooter]['arrows_fired']
        if ready:
            up += dt; arrows_up += fired
        else:
            down += dt; arrows_down += fired
    return dict(target_dash_down_ticks=down, target_dash_up_ticks=up, arrows_while_down=arrows_down, arrows_while_up=arrows_up)


def extras(result, frames, bot_rows, bot, foe):
    """Per-game behaviour metrics for both players (used across every bucket)."""
    out = dict(bot_combos=combos(frames, bot), foe_combos=combos(frames, foe),
               bot_vs_foe_dash=cooldown_windows(frames, bot, foe), foe_vs_bot_dash=cooldown_windows(frames, foe, bot))
    # Bot platform activations split by whether the bot was grounded at the press decision (its own log).
    grounded = airborne = 0
    free = [r for r in bot_rows if not r['latched']]
    for tick in event_ticks(frames, bot, 'platform_activations'):
        press = [r for r in free if r['t'] <= tick and r['buttons'] & 8]
        if press and press[-1]['grounded']:
            grounded += 1
        else:
            airborne += 1
    out['bot_platform_grounded'] = grounded
    out['bot_platform_airborne'] = airborne
    return out


def wall_sdf(e, x, y):
    """Signed distance to a rounded-rect entity (bopl_bots.h bot_wall_sdf; rx/ry, ux/uy carry the half extents)."""
    px, py = x - e['x'], y - e['y']
    fex, fey = math.hypot(e['rx'], e['ry']), math.hypot(e['ux'], e['uy'])
    rx, ry = (e['rx'] / fex, e['ry'] / fex) if fex > 1e-6 else (1.0, 0.0)
    ux, uy = (e['ux'] / fey, e['uy'] / fey) if fey > 1e-6 else (-ry, rx)
    lx, ly = px * rx + py * ry, px * ux + py * uy
    qx, qy = abs(lx) - fex, abs(ly) - fey
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - e['radius']


def platform_owners(frames):
    """id -> (owner player index, spawn tick) for platforms spawned during the game (a PlatformTransform
    morph spawns its StickyRect at the morph body's position, PlatformTransform.CreatePlatform)."""
    seen = {e['id'] for e in frames[0]['entities'] if e['kind'] == 2}
    owners = {}
    for f in frames[1:]:
        for e in f['entities']:
            if e['kind'] != 2 or e['id'] in seen:
                continue
            seen.add(e['id'])
            best = None
            for i, slime in enumerate(f['slimes']):
                if slime['ability_flags'][2] & 8:
                    d = math.hypot(f['players'][i]['x'] - e['x'], f['players'][i]['y'] - e['y'])
                    if best is None or d < best[0]:
                        best = (d, i)
            owners[e['id']] = (best[1] if best and best[0] < 4.0 else None, f['t'])
    return owners


def squish_detail(frames, death_index, dead, owners):
    """Which platform overlapped the victim at death, whose it was, and the escape window the victim had."""
    before, at = frames[death_index - 1], frames[death_index]
    p = at['players'][dead]
    overlapping = sorted(((wall_sdf(e, p['x'], p['y']), e) for e in at['entities'] if e['kind'] == 2), key=lambda t: t[0])
    if not overlapping:
        return dict(sub='no-platform')
    d, e = overlapping[0]
    owner, spawn_tick = owners.get(e['id'], (None, None))
    foe = 1 - dead
    foe_morphing = len(before['slimes']) > foe and bool(before['slimes'][foe]['ability_flags'][2] & 8)
    victim_morphing = len(before['slimes']) > dead and bool(before['slimes'][dead]['ability_flags'][2] & 8)
    detail = dict(platform_id=e['id'], platform_sdf=round(d, 2), platform_owner=owner, platform_spawn_tick=spawn_tick,
                  platform_age_ticks=(at['t'] - spawn_tick) if spawn_tick is not None else None,
                  foe_morphing=foe_morphing, victim_morphing=victim_morphing)
    if spawn_tick is None:
        sub = 'map-platform'
    elif owner == foe:
        sub = 'foe-platform-growing' if foe_morphing else 'foe-platform-settled'
    elif owner == dead:
        sub = 'own-platform-spawn' if victim_morphing else 'own-platform-settled'
    else:
        sub = 'spawn-unattributed'
    if owner == foe:
        # Escape window: from the foe's platform morph start (its anticipation is visible) to the death.
        start = None
        for f in reversed(frames[:death_index]):
            if len(f['slimes']) > foe and f['slimes'][foe]['ability_flags'][2] & 8:
                start = f['t']
            elif start is not None:
                break
        detail['foe_morph_start_tick'] = start
        detail['escape_window_ticks'] = (at['t'] - start) if start is not None else None
        # Distance from the victim to the foe's morph position when the morph began (how close the foe planted it).
        if start is not None:
            f0 = next(f for f in frames if f['t'] == start)
            detail['plant_distance'] = round(math.hypot(f0['players'][foe]['x'] - f0['players'][dead]['x'],
                                                        f0['players'][foe]['y'] - f0['players'][dead]['y']), 2)
    detail['sub'] = sub
    return detail


def classify_game(result, out):
    """Terminal cause of one game, mechanically from the saved frames + bot log."""
    games = out / 'games'
    with gzip.open(games / result['visual'], 'rt') as handle:
        visual = json.load(handle)
    frames = visual['frames']
    bot = result['bot_index']
    foe = 1 - bot
    logs = parse_log(games / result['bot_log'])
    bot_rows = [r for r in logs if r['player'] == bot]
    info = dict(name=result['name'], bot=result['bot'], opponent=result['opponent'], kind=result['kind'],
                map=result['map'], seed=result['seed'], side=result['side'], ticks=result['ticks'],
                timeout=result['timeout'], bot_won=result['bot_won'])
    tel = result['telemetry']
    info['bot_arrows'] = tel[bot]['arrows_fired']
    info['foe_arrows'] = tel[foe]['arrows_fired']
    info['bot_hits'] = tel[bot]['arrow_hits']
    info['bot_dash'] = tel[bot]['dash_activations']
    info['bot_platform'] = tel[bot]['platform_activations']
    info['bot_bow'] = tel[bot]['bow_activations']
    info.update(extras(result, frames, bot_rows, bot, foe))

    # Policy movement: fraction of decisions the foe moved < 0.15 u (the bot's own passivity criterion).
    still = moved = 0
    for a, b in zip(frames, frames[1:]):
        pa, pb = a['players'][foe], b['players'][foe]
        if not (pa['flags'] & 1 and pb['flags'] & 1):
            continue
        if math.hypot(pb['x'] - pa['x'], pb['y'] - pa['y']) < 0.15:
            still += 1
        else:
            moved += 1
    info['foe_still_fraction'] = still / max(1, still + moved)
    bstill = bmoved = 0
    for a, b in zip(frames, frames[1:]):
        pa, pb = a['players'][bot], b['players'][bot]
        if not (pa['flags'] & 1 and pb['flags'] & 1):
            continue
        if math.hypot(pb['x'] - pa['x'], pb['y'] - pa['y']) < 0.15:
            bstill += 1
        else:
            bmoved += 1
    info['bot_still_fraction'] = bstill / max(1, bstill + bmoved)

    if result['timeout']:
        info['bucket'] = 'timeout'
        free = [r for r in bot_rows if not r['latched']]
        info['spot_found_fraction'] = (sum(r['spot'] for r in free) / len(free)) if free else None
        # Oscillation: how often the bot reversed horizontal walking direction.
        xs = [r['x'] for r in free]
        reversals = 0
        last = 0
        for a, b in zip(xs, xs[1:]):
            d = b - a
            if abs(d) < 0.05:
                continue
            s = 1 if d > 0 else -1
            if last and s != last:
                reversals += 1
            last = s
        info['bot_direction_reversals'] = reversals
        info['bot_x_range'] = (max(xs) - min(xs)) if xs else 0
        if info['foe_still_fraction'] > 0.6:
            sub = 'policy-camping'
        elif info['bot_arrows'] == 0:
            sub = 'bot-never-fired'
        else:
            sub = 'both-active'
        if free and info['spot_found_fraction'] is not None and info['spot_found_fraction'] < 0.2 and info['bot_arrows'] < 3:
            sub += '/no-firing-spot'
        info['sub'] = sub
        return info

    # Who died, when, how.
    dead = bot if not result['bot_won'] else foe
    both_dead = not (frames[-1]['players'][0]['flags'] & 1) and not (frames[-1]['players'][1]['flags'] & 1)
    info['double_ko'] = both_dead
    death_index = next(i for i, f in enumerate(frames) if not f['players'][dead]['flags'] & 1)
    before = frames[death_index - 1]
    at = frames[death_index]
    info['death_tick'] = at['t']
    info['death_index'] = death_index
    pd = at['players'][dead]
    td = tel[dead]
    info['cause_code'] = pd['cause_of_death']
    info['death_x'], info['death_y'] = pd['x'], pd['y']
    slime_before = before['slimes'][dead]
    morph_slot = next((s for s in range(3) if slime_before['ability_flags'][s] & 8), None)
    morph = {0: 'dash', 1: 'bow', 2: 'platform', None: None}[morph_slot]
    info['morph_at_death'] = morph
    if td['deaths_environment']:
        if pd['cause_of_death'] == 1:
            cause = 'squished'
        elif pd['y'] < WATER_Y + 1.0 or before['players'][dead]['y'] < WATER_Y + 3 and pd['y'] < 0:
            cause = 'water'
        elif abs(pd['x']) > BLAST_X - 5:
            cause = 'blast-x'
        elif pd['y'] > BLAST_Y - 5:
            cause = 'blast-y'
        else:
            cause = 'environment-other'
    elif td['deaths_self']:
        cause = 'own-arrow'
    elif td['deaths_opponent']:
        cause = 'opponent-arrow'
    else:
        cause = 'other'
    info['cause'] = cause

    if cause == 'squished':
        detail = squish_detail(frames, death_index, dead, platform_owners(frames))
        info.update({('squish_' + k if k != 'sub' else k): v for k, v in detail.items()})
        if dead == bot:
            # What the bot was doing while the foe's platform grew: its decisions inside the escape window.
            window = detail.get('escape_window_ticks') or 0
            rows_in = [r for r in bot_rows if at['t'] - window <= r['t'] <= at['t']]
            info['bot_decisions_in_window'] = len(rows_in)
            info['bot_moved_in_window'] = round(sum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(rows_in, rows_in[1:])), 2)
            info['bot_buttons_in_window'] = sorted({n for r in rows_in if not r['latched'] for bit, n in ((1, 'jump'), (2, 'dash'), (4, 'arrow'), (8, 'platform')) if r['buttons'] & bit})
            info['bot_latched_in_window'] = sum(1 for r in rows_in if r['latched'])
            info['bot_grounded_at_death'] = bool(rows_in and rows_in[-1]['grounded'])
        info['bucket'] = 'squished' if dead == bot else 'bot-win-squish'
        return info

    # What the dead player was doing in the last decisions (for falls).
    last_rows = [r for r in bot_rows if r['t'] <= at['t']][-12:] if dead == bot else []
    recent_buttons = 0
    for r in last_rows:
        if not r['latched']:
            recent_buttons |= r['buttons']
    if cause in ('water', 'blast-x', 'blast-y'):
        # Which action launched the fall: look back from death for the last grounded decision and what followed.
        launch = 'unknown'
        if dead == bot and bot_rows:
            grounded_rows = [r for r in bot_rows if r['t'] <= at['t'] and r['grounded']]
            last_ground = grounded_rows[-1] if grounded_rows else None
            after = [r for r in bot_rows if last_ground and r['t'] >= last_ground['t'] and r['t'] <= at['t']]
            acts = []
            for r in after:
                if r['latched']:
                    continue
                if r['buttons'] & 2: acts.append('dash')
                if r['buttons'] & 8: acts.append('platform')
                if r['buttons'] & 1: acts.append('jump')
            # Ability telemetry over the fall is more reliable than button presses (vetoes strip buttons).
            fall_start = last_ground['t'] if last_ground else max(0, at['t'] - 240)
            f0 = next(f for f in frames if f['t'] >= fall_start)
            d_dash = tel[dead]['dash_activations'] - f0['telemetry'][dead]['dash_activations']
            d_plat = tel[dead]['platform_activations'] - f0['telemetry'][dead]['platform_activations']
            info['fall_dash_uses'] = d_dash
            info['fall_platform_uses'] = d_plat
            info['fall_ticks'] = at['t'] - fall_start
            if last_ground is None:
                launch = 'never-grounded'
            elif acts and acts[0] == 'dash':
                launch = 'dash'
            elif acts and acts[0] == 'platform':
                launch = 'platform-ejection'
            elif acts and acts[0] == 'jump':
                launch = 'jump'
            elif d_plat > 0 and last_ground['morphing'] == 0 and last_ground.get('dodge') == 3:
                launch = 'platform-ejection'
            else:
                # Walked off / pushed / platform sank; check whether the last grounded decision was a dodge.
                launch = 'walk-or-drop'
                if last_ground.get('dodge') and last_ground['dodge'] > 0:
                    launch = 'dodge-' + {1: 'dash', 2: 'jump', 3: 'platform', 4: 'steer'}[last_ground['dodge']]
            # Was the fall a dodge?
            dodge_rows = [r for r in after if not r['latched'] and r.get('dodge') and r['dodge'] > 0]
            info['fall_was_dodge'] = bool(dodge_rows) or (last_ground is not None and last_ground.get('dodge', 0) and last_ground['dodge'] > 0)
            if last_ground is not None:
                # Resources available when the fall began.
                info['fall_resources'] = dict(dash=last_ground['dash'], platform=last_ground['plat'], bow=last_ground['bow'])
                info['fall_from_y'] = last_ground['y']
        info['launch'] = launch
        info['bucket'] = 'fell-' + cause
        info['sub'] = launch
        return info

    if cause in ('own-arrow', 'opponent-arrow'):
        # Identify the killing arrow: the arrow in the frame before death closest to the victim, whose owner matches.
        victim_id = before['players'][dead]['id']
        killer_id = victim_id if cause == 'own-arrow' else at['players'][1 - dead]['id']
        vx, vy = before['players'][dead]['x'], before['players'][dead]['y']
        # Use the morph body position when morphing (player x/y is the slime record).
        cand = [a for a in arrows_in(before) if a['owner'] == killer_id]
        if not cand:  # fired and landed inside the same 4-tick step: the arrow only exists in the death frame
            cand = [a for a in arrows_in(at) if a['owner'] == killer_id]
            vx, vy = at['players'][dead]['x'], at['players'][dead]['y']
            info['point_blank'] = True
        if cand:
            k = min(cand, key=lambda a: math.hypot(a['x'] - vx, a['y'] - vy))
            info['killer_arrow'] = dict(x=k['x'], y=k['y'], vx=k['vx'], vy=k['vy'], speed=math.hypot(k['vx'], k['vy']),
                                        distance=math.hypot(k['x'] - vx, k['y'] - vy))
            # Arrow age: first frame in which an arrow with this id existed.
            born = next((f['t'] for f in frames if any(e['id'] == k['id'] for e in arrows_in(f))), None)
            info['arrow_age_ticks'] = at['t'] - born if born is not None else None
            info['arrow_launch_range'] = None
            if born is not None:
                bf = next(f for f in frames if f['t'] == born)
                ke = next(e for e in arrows_in(bf) if e['id'] == k['id'])
                p = bf['players'][dead]
                info['arrow_launch_range'] = math.hypot(ke['x'] - p['x'], ke['y'] - p['y'])
        if dead != bot:
            info['bucket'] = 'bot-win-arrow' if cause == 'opponent-arrow' else 'policy-own-arrow'
            info['sub'] = morph or ('airborne' if before['slimes'][dead]['state'] in (2, 8) else 'grounded')
            return info
        # Bot died to an arrow: what did its dodge layer see?
        window_rows = [r for r in bot_rows if at['t'] - 120 <= r['t'] <= at['t']]
        min_eta = min((r['eta'] for r in window_rows), default=1e9)
        saw = [r for r in window_rows if r['eta'] <= DODGE_WINDOW_HARD]
        acted = [r for r in saw if not r['latched'] and r.get('dodge') and r['dodge'] > 0]
        latched_saw = [r for r in saw if r['latched']]
        vetoed = [r for r in saw if not r['latched'] and r.get('veto')]
        info['min_eta_seen'] = min_eta
        info['dodge_decisions'] = len(acted)
        info['dodge_kinds'] = sorted({{1: 'dash', 2: 'jump', 3: 'platform', 4: 'steer'}[r['dodge']] for r in acted})
        info['vetoed_decisions'] = len(vetoed)
        info['latched_while_threatened'] = len(latched_saw)
        info['morphing_while_threatened'] = sum(1 for r in saw if r['morphing'])
        first_seen = saw[0]['t'] if saw else None
        info['threat_lead_ticks'] = (at['t'] - first_seen) if first_seen is not None else 0
        if cause == 'own-arrow':
            info['bucket'] = 'own-arrow'
            # Shoot -> dash: the last dash decision before death vs. the killing arrow's birth (Matthew's hypothesis:
            # the dash is chosen while the arrow is still inside its fire delay, so bot_dash_safe never saw it).
            free = [r for r in bot_rows if not r['latched'] and r['t'] <= at['t']]
            dash_rows = [r for r in free if r['buttons'] & 2]
            last_dash = dash_rows[-1] if dash_rows else None
            born = at['t'] - info['arrow_age_ticks'] if info.get('arrow_age_ticks') is not None else None
            if last_dash is not None:
                info['dash_decision_tick'] = last_dash['t']
                info['dash_to_death_ticks'] = at['t'] - last_dash['t']
                info['arrows_live_at_dash_decision'] = last_dash['arrows']
                if born is not None:
                    # >0: the arrow already existed when the dash was chosen (dash_safe could have seen it)
                    info['arrow_age_at_dash_decision'] = last_dash['t'] - born
            # Last arrow press (release decision) before death.
            press_rows = [r for r in free if r['buttons'] & 4]
            if press_rows and last_dash is not None:
                info['arrow_press_to_dash_ticks'] = last_dash['t'] - press_rows[-1]['t']
            dashing_now = morph == 'dash' or (last_dash is not None and at['t'] - last_dash['t'] <= 24)
            if dashing_now:
                if info.get('arrow_age_at_dash_decision') is not None and info['arrow_age_at_dash_decision'] < 0:
                    sub = 'shoot-then-dash-into-own-arrow/arrow-not-yet-live-at-dash'
                elif last_dash is not None and last_dash['arrows'] == 0:
                    sub = 'shoot-then-dash-into-own-arrow/arrow-not-in-live-list'
                else:
                    sub = 'dash-into-own-arrow/arrow-was-live'
            elif morph == 'platform':
                sub = 'platform-ejection-into-own-arrow'
            elif morph == 'bow':
                sub = 'bow-morph-hit-by-own-arrow'
            elif recent_buttons & 1 and info.get('arrow_age_ticks') is not None and info['arrow_age_ticks'] < 40:
                sub = 'jump-into-own-fresh-arrow'
            elif (info.get('arrow_age_ticks') or 0) > 60:
                sub = 'own-mortar-returning'
            else:
                sub = 'own-arrow-fresh-other'
            info['sub'] = sub
            return info
        # Opponent arrow.
        info['bucket'] = 'hit-by-arrow'
        if morph in ('bow', 'platform'):
            sub = f'killed-while-morphing-{morph}'
        elif morph == 'dash':
            sub = 'killed-mid-dash'
        elif not saw:
            age = info.get('arrow_age_ticks')
            if age is not None and age <= 8:
                sub = 'point-blank-no-dodge-possible'   # arrow flew < 2 decisions
            elif window_rows and all(r['latched'] or r['morphing'] for r in window_rows if at['t'] - r['t'] <= (age or 30)):
                sub = 'latched-or-morphing-through-flight'
            else:
                sub = 'threat-never-predicted'         # arrow flew >= 2 decisions and bot_arrow_eta never saw it
        elif acted:
            kinds = info['dodge_kinds']
            # Second arrow: another live enemy arrow besides the killer was in flight when the dodge was chosen.
            enemy_arrows = {e['id'] for f in frames[max(0, death_index - 8):death_index] for e in arrows_in(f) if e['owner'] == killer_id}
            sub = 'dodged-into-second-arrow' if len(enemy_arrows) > 1 else 'dodge-too-slow-' + '+'.join(kinds)
        elif latched_saw and not any(not r['latched'] for r in saw):
            sub = 'latched-could-not-dodge'
        elif vetoed:
            sub = 'dodge-vetoed'
        else:
            sub = 'threat-seen-no-dodge-action'
        info['sub'] = sub
        return info

    info['bucket'] = cause
    info['sub'] = morph or ''
    return info


def classify(args):
    out = args.out.resolve()
    schedule=json.loads((out/'schedule.json').read_text()) if (out/'schedule.json').is_file() else {}
    catalogue_records=[schedule]
    for checkpoint in (schedule.get('checkpoints') or {}).values():
        try:
            catalogue_records.append(json.loads(Path(checkpoint+'.json').read_text()))
        except (OSError,ValueError):
            pass
    rows = [normalize_record(json.loads(l),*catalogue_records)
            for l in (out / 'games.jsonl').read_text().splitlines() if l.strip()]
    seen = set(); unique = []
    for r in rows:
        if r['name'] not in seen:
            seen.add(r['name']); unique.append(r)
    infos = []
    for i, r in enumerate(unique):
        try:
            infos.append(classify_game(r, out))
        except Exception as error:  # keep going; report the failure
            infos.append(dict(name=r['name'], bot=r['bot'], opponent=r['opponent'], kind=r['kind'], bucket='classify-error',
                              sub=repr(error), timeout=r['timeout'], bot_won=r['bot_won'], ticks=r['ticks']))
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(unique)}', flush=True)
    (out / 'taxonomy.json').write_text(json.dumps(infos, indent=0))
    print(f'wrote {out / "taxonomy.json"} ({len(infos)} games)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('play')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--bots', default='archer,pursuer,evasive,mixed')
    p.add_argument('--maps', default='0,1,2')
    p.add_argument('--seed-pairs', type=int, default=7)
    p.add_argument('--bot-vs-bot-pairs', type=int, default=4)
    p.add_argument('--workers', type=int, default=6)
    p.set_defaults(func=play)
    c = sub.add_parser('classify')
    c.add_argument('--out', type=Path, required=True)
    c.set_defaults(func=classify)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
