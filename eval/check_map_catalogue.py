"""Per-map smoke of the Rust sim's level catalogue.

For every RL map index (default: all the backend reports) and a few seeds:

  dry      after the warmup and two idle seconds both players are alive, on the
           ground and above the water line (spawns are not in the water)
  settle   the level's platforms (kind 2, wall layers) have finished animating in:
           positions at 3 s and 5 s agree to 1e-3 (orbit platforms excluded)
  random   a full episode of random legal inputs against a scripted bot runs to
           its terminal without a game error, every v3 and v4 observation row is
           finite, and the v4 ray block stays in [0, 1]

and a `digest`: the sha256 of every observation row of the run, so two builds of
the sim can be compared bit for bit on the maps they share (--digest-only).

Usage:
  check_map_catalogue.py [--maps 5,6,...] [--seeds 3] [--game LIB] [--json OUT]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import layout

ROOT = Path(__file__).resolve().parents[1]
WATER_Y = -18.95            # WaterHitBox on every grass level
WALL_LAYERS = (11, 16, 21)  # bopl_obs_v4.h: the platform layers the writer treats as walls
PAIRINGS = []               # (kind, level) of the bots that face the --bot on odd seeds


def fix(v):
    return v / 4294967296.


def sample_masked(rng, mask):
    out, lo = [], 0
    for size in layout.ACTION_SIZES:
        allowed = np.flatnonzero(mask[lo:lo + size])
        out.append(int(rng.choice(allowed)) if allowed.size else 0)
        lo += size
    return out


def platforms(env):
    s = env.state
    return {e.id: (fix(e.x), fix(e.y)) for e in s.entities[:s.header.entity_count]
            if e.kind == 2 and e.layer in WALL_LAYERS and e.flags & 1 and not e.flags & 4}


def idle(env, decisions):
    for _ in range(decisions):
        env.step([[0] * layout.NUM_ATNS, [0] * layout.NUM_ATNS])
        if env.ended:
            return


def check_map(Environment, map_id, seeds, bot, digest, orbits, lengths, causes):
    problems = []
    for seed in seeds:
        with Environment(seed=seed, map_id=map_id, limit=3600, delay=9) as env:
            env.set_layout(0, layout.V4_LAYOUT_VERSION)
            digest.update(env.rows.tobytes())
            # dry and settle: nobody plays; the level animated in during the warmup,
            # so after two idle seconds both slimes stand and no platform moves
            # (orbit platforms start their circuit after their authored delay).
            at0 = platforms(env)
            idle(env, 30)
            digest.update(env.rows.tobytes())
            f = env.frame()
            for p in f['players']:
                if p['deaths'] or p['y'] < WATER_Y + 0.5:
                    problems.append(f"seed {seed}: player {p['id']} not dry after idle (deaths {p['deaths']}, y {p['y']:.2f})")
            for sl in f['slimes']:
                if abs(sl['vy']) > 0.05:
                    problems.append(f"seed {seed}: slime {sl['owner']} still falling after idle (vy {sl['vy']:.3f})")
            at2 = platforms(env)
            moving = [pid for pid in at0 if pid in at2 and max(abs(a - b) for a, b in zip(at0[pid], at2[pid])) > 1e-3]
            if len(moving) > orbits:
                problems.append(f"seed {seed}: {len(moving)} platform(s) moved in the first two seconds, {orbits} orbit(s) authored: {sorted(moving)}")
            if not at2:
                problems.append(f"seed {seed}: no platforms in the state")
            # random: the episode played out against the bot; odd seeds pit two bots
            # against each other instead, which lives long enough to cover the level.
            env.reset(seed, map_id)
            env.bot(1, *bot)
            scripted = seed % 2 == 1
            if scripted:
                # bots are deterministic from the state, so vary the pairing, not the seed
                env.bot(0, *PAIRINGS[(seed // 2) % len(PAIRINGS)])
            rng = np.random.default_rng(seed)
            ticks = 0
            while not env.ended and ticks < 3600 + 8:
                human = [0] * layout.NUM_ATNS if scripted else sample_masked(rng, env.mask[0])
                env.step([human, [0] * layout.NUM_ATNS])
                ticks = env.state.header.episode_ticks
                rows = env.rows
                digest.update(rows.tobytes())
                if not np.isfinite(rows).all():
                    problems.append(f"seed {seed}: non-finite observation at tick {ticks}"); break
                rays = env.obs_row(0)[layout.V4_RAY_OFFSET:layout.V4_RAY_OFFSET + layout.V4_RAYS * layout.V4_RAY_FEATURES]
                dist = rays.reshape(layout.V4_RAYS, layout.V4_RAY_FEATURES)[:, 0]
                if dist.min() < 0 or dist.max() > 1:
                    problems.append(f"seed {seed}: v4 ray distance outside [0,1] at tick {ticks}: {dist.min():.3f}..{dist.max():.3f}"); break
            if not env.ended:
                problems.append(f"seed {seed}: episode did not end within {ticks} ticks")
            f = env.frame()
            digest.update(json.dumps(f['players'], sort_keys=True).encode())
            lengths.append(ticks)
            for t in f['telemetry']:
                for k in ('deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other'):
                    causes[k] = causes.get(k, 0) + t[k]
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maps', help='RL map indices (default: every map the backend reports)')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--seed', type=int, default=7300)
    p.add_argument('--game', help='sim library (default: the selected BOPL_RL_GAME / native/target/release/libbopl_sim.so)')
    p.add_argument('--bot', default='mixed-v2-hard')
    p.add_argument('--json', help='write the report here')
    p.add_argument('--digest-only', action='store_true', help='report only the per-map digests (build comparison)')
    a = p.parse_args()
    from environment import Environment, configure, LIB
    from evaluation_protocol import scripted_opponent
    import ctypes as C
    configure(Path('/tmp/bopl-check-maps-logs'), a.game or str(ROOT / 'native/target/release/libbopl_sim.so'))
    bot = scripted_opponent(a.bot)
    PAIRINGS.extend(scripted_opponent(n) for n in ('archer-v2-hard', 'pursuer-v2-hard', 'evasive-v2-hard', 'mixed-hard'))
    if a.maps:
        maps = [int(m) for m in a.maps.split(',')]
    else:
        # bopl_rl_map_count through a throwaway environment: the research library
        # opens the game lazily, so ask the sim directly.
        sim = C.CDLL(a.game or str(ROOT / 'native/target/release/libbopl_sim.so'))
        sim.bopl_rl_map_count.restype = C.c_uint32
        maps = list(range(sim.bopl_rl_map_count(None)))
    table = {}
    table_path = ROOT / 'native/data/map-set.json'
    if table_path.exists():
        table = {m['rl']: m for m in json.loads(table_path.read_text())['maps']}
    rollout = json.loads((ROOT / 'native/data/rollout.json').read_text())['maps']
    orbits = [sum('orbit' in p for p in m['platforms']) for m in rollout]
    report = {'maps': [], 'ok': True}
    for map_id in maps:
        digest = hashlib.sha256()
        seeds = [a.seed + 10 * map_id + i for i in range(a.seeds)]
        lengths, causes = [], {}
        problems = check_map(Environment, map_id, seeds, bot, digest, orbits[map_id] if map_id < len(orbits) else 0, lengths, causes)
        entry = {'map': map_id, 'level': table.get(map_id, {}).get('level'), 'seeds': seeds,
                 'digest': digest.hexdigest(), 'episode_ticks': lengths, 'deaths': causes, 'problems': problems}
        report['maps'].append(entry)
        if problems and not a.digest_only:
            report['ok'] = False
        line = f"map {map_id:2d} level {str(entry['level']):>3}: {entry['digest'][:16]} ticks {lengths} deaths env/opp/self {causes.get('deaths_environment', 0)}/{causes.get('deaths_opponent', 0)}/{causes.get('deaths_self', 0)}"
        if not a.digest_only:
            line += '  ok' if not problems else '  ' + '; '.join(problems)
        print(line, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
