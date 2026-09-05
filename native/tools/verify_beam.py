"""Exact beam geometry, color and graph checks against the actual Unity methods."""
import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
from collections import Counter

from beam_cases import graph_queries, queries
from reference import ROOT, first_difference

SCHEMA = "bopl-beam-golden-v1"


def encoded(records):
    return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(records):
    result = subprocess.run(
        [str(ROOT / "native/target/release/dynamics-probe")],
        input=encoded(r["query"] for r in records), text=True,
        capture_output=True, check=True,
    )
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if len(outputs) != len(records):
        raise ValueError("Incomplete Rust beam corpus")
    counts = Counter()
    roles = Counter()
    failures = []
    for i, (record, actual) in enumerate(zip(records, outputs)):
        expected = record["expected"]
        if isinstance(expected, dict):
            counts["hits"] += expected.get("hit", False)
            counts["source_errors"] += "error" in expected
        else:
            counts["frames"] += len(expected)
            for frame in expected:
                counts["reflections"] += sum(frame["reflected"])
                counts["crossings"] += len(frame["crossings"])
                counts["hit_targets"] += sum(hit is not None for hit in frame["hits"])
                roles.update(frame["roles"])
        if actual != expected:
            counts["mismatches"] += 1
            if len(failures) < 10:
                failures.append({"case": i, "difference": first_difference(expected, actual)})
    return {
        "match": not counts["mismatches"], "cases": len(records),
        **{key: counts[key] for key in ("hits", "frames", "reflections", "crossings",
                                       "hit_targets", "source_errors", "mismatches")},
        "roles": dict(roles), "first_failures": failures,
    }


def reference(directory, inputs, reuse):
    directory.mkdir(exist_ok=True, parents=True)
    source = directory / "input.jsonl"
    output = directory / "unity.jsonl"
    provenance = directory / "reference-sha256.json"
    raw = encoded(inputs)
    if reuse:
        if source.read_text() != raw:
            raise ValueError("Cannot reuse changed beam queries")
    else:
        source.write_text(raw)
        env = dict(os.environ, BOPL_KERNEL_INPUT=str(source), BOPL_KERNEL_OUTPUT=str(output))
        for key in ("BOPL_TRACE", "BOPL_TRACE_TICKS", "BOPL_TRACE_BATCH"):
            env.pop(key, None)
        log = directory / "unity.log"
        with (directory / "stdout.log").open("w") as stdout:
            subprocess.run(
                [str(ROOT / "Builds/Linux/BoplBattle.x86_64"),
                 str(ROOT / "native/fixtures/movement.rep"), "-batchmode", "-nographics",
                 "-logFile", str(log)],
                env=env, stdout=stdout, stderr=subprocess.STDOUT, check=True, timeout=600,
            )
        if "REFERENCE_KERNEL_COMPLETE" not in log.read_text():
            raise ValueError("Incomplete Unity beam probe")
        provenance.write_text(json.dumps({
            "assembly": digest(ROOT / "Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll"),
            "input": digest(source), "output": digest(output),
        }))
    hashes = json.loads(provenance.read_text())
    if hashes["input"] != digest(source) or hashes["output"] != digest(output):
        raise ValueError("Beam reference provenance mismatch")
    expected = [json.loads(line) for line in output.read_text().splitlines()]
    if len(expected) != len(inputs):
        raise ValueError("Incomplete beam reference")
    return [{"query": q, "expected": e} for q, e in zip(inputs, expected)], hashes


def promote(fixture, records, hashes, seed, graph):
    first, extra = (12, 48) if graph else (300, 600)
    selected = set(range(min(first, len(records))))
    selected.update(random.Random(seed).sample(range(len(records)), min(extra, len(records))))
    # Keep source failures, all fixed color edges and the first instance of each graph role.
    seen_roles = set()
    for i, record in enumerate(records):
        expected = record["expected"]
        if isinstance(expected, dict):
            if "error" in expected or (not graph and i >= len(records) - 100):
                selected.add(i)
        else:
            roles = {role for frame in expected for role in frame["roles"]}
            if roles - seen_roles:
                selected.add(i)
                seen_roles.update(roles)
    header = {"schema": SCHEMA, "cases": len(selected), "provenance": hashes}
    raw = encoded([header, *[records[i] for i in sorted(selected)]])
    fixture.write_bytes(gzip.compress(raw.encode(), mtime=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=641924)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--stress", action="store_true", help="120-frame graphs with three beams per player")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--reuse-unity", action="store_true")
    args = parser.parse_args()
    args.graph |= args.stress
    name = "beam-graph-stress" if args.stress else "beam-graph" if args.graph else "beam"
    fixture = ROOT / f"native/fixtures/{name}.jsonl.gz"
    if args.golden:
        with gzip.open(fixture, "rt") as stream:
            header = json.loads(next(stream))
            records = [json.loads(line) for line in stream]
        if header["schema"] != SCHEMA or len(records) != header["cases"]:
            raise ValueError("Invalid beam golden")
        report = check(records)
    else:
        if args.cases < 1:
            parser.error("--cases must be positive")
        directory = ROOT / f"native/artifacts/{name}"
        if args.graph:
            inputs = list(graph_queries(args.cases, args.seed,
                                       copies=3 if args.stress else 1,
                                       ticks=120 if args.stress else 12))
        else:
            inputs = list(queries(args.cases, args.seed))
        records, hashes = reference(directory, inputs, args.reuse_unity)
        report = {**check(records), "provenance": hashes}
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["match"]:
            promote(fixture, records, hashes, args.seed, args.graph)
    print(json.dumps(report, indent=2))
    return int(not report["match"])


if __name__ == "__main__":
    raise SystemExit(main())
