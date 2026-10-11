"""Re-step lossless human replays in the sim and dump every tick's full ABI
state (players, slimes, entities, telemetry, inputs) as numpy tables, one .npz
per replay, for offline analysis.

  python eval/replay_dump.py native/artifacts/human-replays --out native/artifacts/human-replays/dump --jobs 6

Fixed-point Q32.32 values are converted to float32 (world units, seconds).
Tables are long-form with a `tick` column; see FIELDS below and the ABI
(sim/abi/bopl_rl.h) for meanings. Entity `kind`: 0 circle, 1 box, 2 rounded
rect, 3 rope segment, 4 beam. Entity `source_ability` is the ABI's reserved[0]
(the ability that spawned or granted it), `timer` its reserved[1] (ticks to the
next event when flags & 64). Ability names: native/data/ability-coverage.json.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_data import ReplayWorld, decode

Q = 1.0 / (1 << 32)
PLAYER = [('tick', 'u4'), ('player', 'u1'), ('id', 'u4'), ('team', 'u4'), ('flags', 'u4'), ('alive_bodies', 'u4'),
          ('stocks', 'u4'), ('kills', 'u4'), ('deaths', 'u4'), ('cause_of_death', 'u4'),
          ('x', 'f4'), ('y', 'f4'), ('scale', 'f4'), ('invisible_time', 'f4'), ('ability_time', 'f4')]
SLIME = [('tick', 'u4'), ('id', 'u4'), ('owner', 'u4'), ('flags', 'u4'), ('state', 'u4'),
         ('x', 'f4'), ('y', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('rotation', 'f4'), ('scale', 'f4'),
         ('abilities', 'u4', 3), ('ability_flags', 'u4', 3), ('cooldowns', 'f4', 3), ('timers', 'f4', 3)]
ENTITY = [('tick', 'u4'), ('id', 'u4'), ('kind', 'u1'), ('layer', 'u4'), ('owner', 'u4'), ('flags', 'u4'),
          ('hierarchy', 'u4'), ('source_ability', 'u4'), ('timer', 'u4'),
          ('x', 'f4'), ('y', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('right_x', 'f4'), ('right_y', 'f4'),
          ('up_x', 'f4'), ('up_y', 'f4'), ('radius', 'f4'), ('rotation', 'f4'), ('angular_velocity', 'f4'),
          ('inverse_mass', 'f4'), ('clock_rate', 'f4')]
TELEMETRY = ['ability_activations0', 'ability_activations1', 'ability_activations2', 'arrows_fired', 'arrow_hits',
             'deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other']


def dump(path, out):
    meta, inputs = decode(path)
    target = Path(out) / (Path(path).stem + '.npz')
    if target.exists():
        return str(target), 'cached'
    players, slimes, entities, telemetry, header = [], [], [], [], []
    count = len(meta['players'])
    with ReplayWorld(meta, Path(out) / 'logs' / (Path(path).stem + '.log')) as world:
        for t in range(meta['ticks']):
            world.step(list(zip(inputs[t]['angle'], inputs[t]['buttons'])))
            s = world.state
            h = s.header
            header.append((h.tick, h.episode_ticks, h.flags, h.entity_count, h.slime_count, h.sudden_death))
            for i in range(count):
                p = s.players[i]
                players.append((t, i, p.id, p.team, p.flags, p.alive_bodies, p.stocks, p.kills, p.deaths, p.cause_of_death,
                                p.x * Q, p.y * Q, p.scale * Q, p.invisible_time * Q, p.ability_time * Q))
                # The ctypes Telemetry names its first three fields after the
                # phase-1 kit; they are ability_activations[0..2] in slot order.
                telemetry.append(tuple([t, i] + [getattr(s.telemetry[i], n) for n, _ in s.telemetry[i]._fields_]))
            for k in range(h.slime_count):
                m = s.slimes[k]
                slimes.append((t, m.id, m.owner, m.flags, m.state, m.x * Q, m.y * Q, m.vx * Q, m.vy * Q,
                               m.rotation * Q, m.scale * Q, tuple(m.abilities), tuple(m.ability_flags),
                               tuple(c * Q for c in m.cooldowns), tuple(c * Q for c in m.timers)))
            for k in range(h.entity_count):
                e = s.entities[k]
                entities.append((t, e.id, e.kind, e.layer, e.owner, e.flags, e.hierarchy, e.reserved0, e.reserved1,
                                 e.x * Q, e.y * Q, e.vx * Q, e.vy * Q, e.right_x * Q, e.right_y * Q, e.up_x * Q,
                                 e.up_y * Q, e.radius * Q, e.rotation * Q, e.angular_velocity * Q,
                                 e.inverse_mass * Q, e.clock_rate * Q))
            if world.ended:
                break
    np.savez_compressed(
        target,
        meta=np.array(json.dumps({k: v for k, v in meta.items()})),
        header=np.array(header, dtype=[('tick', 'u4'), ('episode_ticks', 'u4'), ('flags', 'u4'),
                                       ('entity_count', 'u4'), ('slime_count', 'u4'), ('sudden_death', 'u4')]),
        players=np.array(players, dtype=PLAYER),
        slimes=np.array(slimes, dtype=SLIME),
        entities=np.array(entities, dtype=ENTITY),
        telemetry=np.array(telemetry, dtype=[('tick', 'u4'), ('player', 'u1')] + [(f, 'u4') for f in TELEMETRY]),
        inputs=np.ascontiguousarray(inputs[:len(header)]))
    return str(target), len(header)


def job(args):
    try:
        return dump(*args)
    except Exception as error:  # a replay the sim rejects is reported, not fatal
        return args[0], f'error: {error}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('replays', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.replays.glob('*.rep'))
    lossless = []
    for f in files:
        try:
            decode(f)
            lossless.append(f)
        except ValueError:
            pass
    print(f'{len(lossless)} of {len(files)} replays are lossless BOPLREP2', flush=True)
    with Pool(args.jobs) as pool:
        for n, (path, result) in enumerate(pool.imap_unordered(job, [(str(f), str(args.out)) for f in lossless]), 1):
            print(f'{n}/{len(lossless)} {os.path.basename(path)} {result}', flush=True)


if __name__ == '__main__':
    main()
