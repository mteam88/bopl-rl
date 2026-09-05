"""Produce lossless 1..6-player inputs against the pinned ability catalog."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def catalog_identity():
    catalog = json.loads((ROOT / 'native/data/ability-coverage.json').read_text())['abilities']
    assets = json.loads((ROOT / 'native/reference-export/assets.json').read_text())
    identity = '\n'.join(f"{entry['id']}:{entry['name']}:{assets[entry['prefab_guid']]['data']['name'] if entry['prefab_guid'] else ''}" for entry in catalog)
    return catalog, hashlib.sha256(identity.encode()).hexdigest()


def scenario(players, abilities, seed, ticks, level=0, stocks=0, teams=None):
    catalog, digest = catalog_identity()
    rng = random.Random(seed)
    setup = {
        'sourceHash': json.loads((ROOT / 'native/data/full-reference.json').read_text())['content_sha256'],
        'catalogHash': digest, 'seed': seed, 'level': level, 'gameMode': 0, 'stocks': stocks,
        'players': [{'id': p+1, 'color': p, 'team': teams[p] if teams else p, 'keyboard': False, 'abilities': abilities[p]} for p in range(players)],
        'frames': [],
    }
    held = [0] * players
    for tick in range(ticks):
        frame = []
        for p in range(players):
            angle = buttons = 0
            if tick >= 240:
                if tick % 45 == 0:
                    held[p] = rng.choice([0, 1, 32, 65, 97, 128, 160, 192, 224])
                angle = held[p]
                # Separate slots as well as simultaneous casts, holds and releases.
                phase = (tick-240) % 120
                if phase < 12: buttons = 2
                elif phase < 24: buttons = 4
                elif phase < 36: buttons = 8
                elif phase < 40: buttons = 15
                if tick % 67 < 3: buttons |= 1
            frame.append({'seqNumber': tick+1, 'joystickAngle': angle,
                'jump': bool(buttons & 1), 'ab1': bool(buttons & 2), 'ab2': bool(buttons & 4), 'ab3': bool(buttons & 8)})
        setup['frames'].append(frame)
    return setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ticks', type=int, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tool = ROOT / 'native/tools/replay-codec/bin/Release/net8.0/replay-codec.dll'
    cases = []
    for players in range(1, 7):
        cases.append((f'{players}-players-dash', scenario(players, [[2, 2, 2]]*players, 12345, args.ticks)))
    abilities = list(range(1, len(catalog_identity()[0])))
    for offset in range(0, len(abilities), 18):
        ids = (abilities[offset:offset+18] + [2]*18)[:18]
        cases.append((f'6-players-abilities-{offset//18}', scenario(6, [ids[p*3:p*3+3] for p in range(6)], 12345+offset, args.ticks, stocks=5)))
    cases.append(('6-players-teams', scenario(6, [[31, 36, 38]]*6, 67484, args.ticks, stocks=3, teams=[0, 0, 1, 1, 2, 2])))
    for name, case in cases:
        path = args.output / (name+'.json')
        path.write_text(json.dumps(case)+'\n')
        subprocess.run(['dotnet', str(tool), 'encode', str(path), str(path.with_suffix('.rep'))], check=True)
    (args.output/'cases.json').write_text(json.dumps([{'name': name, 'players': len(case['players']), 'abilities': [p['abilities'] for p in case['players']], 'ticks': args.ticks} for name, case in cases], indent=2)+'\n')


if __name__ == '__main__': main()
