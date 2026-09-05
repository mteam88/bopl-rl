"""Capture actual clone creation before scheduler Init; this is source evidence only."""
import argparse
import json
from verify_beam import reference
from reference import ROOT


def queries():
    for count in range(1, 7):
        for mode in ['birth', 'limit', 'over_limit', 'failure', 'failure_at_limit']:
            bodies = {'birth': 1, 'limit': 16, 'over_limit': 18,
                      'failure': 15, 'failure_at_limit': 17}[mode]
            players = [dict(id=i+1, bodies=bodies, revival=i % 4) for i in range(count)]
            actions = []
            for repeat in range(3):
                for i in range(count):
                    actions.append(dict(player=i, position=dict(x=(i*7-15)*2**32+repeat,
                                        y=(repeat*3+2)*2**32-i),
                                        null_prefab=mode.startswith('failure') and repeat == 0))
            yield dict(operation='dynamics_clone_birth', players=players,
                       actions=actions, added=count*8, prefabs_added=count*3,
                       clone_case=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-unity', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'native/artifacts/clone-birth'
    records, hashes = reference(directory, list(queries()), args.reuse_unity)
    counts = dict(cases=len(records), calls=0, births=0, limit_returns=0,
                  failure_prefixes=0, registered_components=0)
    for record in records:
        previous = 0
        for frame in record['expected']:
            counts['calls'] += 1
            counts['births'] += frame['returned']
            counts['failure_prefixes'] += frame['error'] is not None
            counts['limit_returns'] += not frame['returned'] and frame['error'] is None
            added = len(frame['clones']) - previous
            counts['registered_components'] += added * 8
            previous = len(frame['clones'])
            for clone in frame['clones']:
                assert not clone['physics_registered'], 'Collider registered before Init'
                assert clone['material_matches'] and clone['sprite_null']
    report = dict(source_only=True, provenance=hashes, **counts)
    (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
