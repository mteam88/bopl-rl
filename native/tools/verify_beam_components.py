"""Verify fixed curves, beam hitboxes and beam target effects against Unity."""
import argparse
import gzip
import json
import random
import subprocess
from collections import Counter

from animation_cases import curve_queries, hitbox_queries
from beam_effect_cases import queries as effect_queries
from reference import ROOT, first_difference
from verify_beam import encoded, reference

SCHEMA = "bopl-animation-golden-v1"


def check(records):
    result = subprocess.run([str(ROOT / "native/target/release/dynamics-probe")],
                            input=encoded(r["query"] for r in records),
                            text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(records):
        raise ValueError("Incomplete native component corpus")
    errors = values = mismatches = 0
    failures = []
    target_changes = Counter()
    for i, (record, actual) in enumerate(zip(records, outputs)):
        expected = record["expected"]
        if isinstance(expected, list):
            values += len(expected)
            if record["query"]["operation"] == "dynamics_beam_effects":
                previous = record["query"]["targets"]
                for frame in expected:
                    for before, after in zip(previous, frame):
                        if before != after:
                            target_changes[after["kind"]] += 1
                    previous = frame
        else:
            errors += 1
        if actual != expected:
            mismatches += 1
            if len(failures) < 10:
                failures.append({"case": i, "difference": first_difference(expected, actual)})
    return {"match": mismatches == 0, "cases": len(records), "values_or_states": values,
            "source_errors": errors, "mismatches": mismatches, "target_changes": dict(target_changes),
            "first_failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=901314)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--hitboxes", action="store_true")
    mode.add_argument("--effects", action="store_true")
    parser.add_argument("--reuse-unity", action="store_true")
    parser.add_argument("--golden", action="store_true")
    args = parser.parse_args()
    name = "beam-effects" if args.effects else "beam-hitbox" if args.hitboxes else "curves"
    fixture = ROOT / f"native/fixtures/{name}.jsonl.gz"
    if args.golden:
        with gzip.open(fixture, "rt") as stream:
            header = json.loads(next(stream))
            records = [json.loads(line) for line in stream]
        if header["schema"] != SCHEMA or header["cases"] != len(records):
            raise ValueError("Invalid component fixture")
        report = check(records)
    else:
        if args.cases < 1:
            parser.error("--cases must be positive")
        directory = ROOT / f"native/artifacts/{name}"
        generate = effect_queries if args.effects else hitbox_queries if args.hitboxes else curve_queries
        inputs = list(generate(args.cases, args.seed))
        records, hashes = reference(directory, inputs, args.reuse_unity)
        report = {**check(records), "provenance": hashes}
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["match"]:
            selected = set(range(min(25, len(records))))
            selected.update(random.Random(args.seed).sample(range(len(records)), min(175, len(records))))
            selected.update(i for i, r in enumerate(records) if isinstance(r["expected"], dict))
            header = {"schema": SCHEMA, "cases": len(selected), "provenance": hashes}
            fixture.write_bytes(gzip.compress(encoded([header, *[records[i] for i in sorted(selected)]]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report["match"])


if __name__ == "__main__":
    raise SystemExit(main())
