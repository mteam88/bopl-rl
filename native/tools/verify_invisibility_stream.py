"""Long independent Invisibility episodes with bounded-memory exact comparison."""
import argparse
import json
import random
from actor_stream import ROOT, capture, verify, promote, golden
from verify_invisibility import queries, fault_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=24)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=731249)
    parser.add_argument('--faults', action='store_true')
    parser.add_argument('--golden', action='store_true')
    parser.add_argument('--reuse-unity', action='store_true')
    args = parser.parse_args()
    name = 'invisibility-failures-stream' if args.faults else 'invisibility-stream'
    fixture = ROOT / f'native/fixtures/{name}.jsonl.gz'
    if args.golden:
        report = golden(fixture)
    else:
        directory = ROOT / f'native/artifacts/{name}'
        inputs = list(fault_queries(args.seed) if args.faults else queries(args.cases, args.seed, args.steps))
        inputs, hashes = capture(directory, inputs, args.reuse_unity)
        report = dict(verify(directory, inputs), provenance=hashes)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['match']:
            selected = set(range(min(6, len(inputs)))) | set(random.Random(args.seed).sample(range(len(inputs)), min(6, len(inputs)))) | set(report['error_cases'])
            promote(directory, fixture, inputs, selected, hashes,
                    'Full-frame independent scheduled Invisibility casts and cooldown recovery; supplied-state initialization, headless indicators, no deaths or actor ropes')
    print(json.dumps(report, indent=2))
    return int(not report['match'])


if __name__ == '__main__':
    raise SystemExit(main())
