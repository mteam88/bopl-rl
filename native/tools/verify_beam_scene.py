"""Compare the combined beam/physics stage and queued callbacks with Unity."""
import argparse
import gzip
import json
import os
import random
import subprocess
from collections import Counter

from beam_scene_cases import queries
from reference import ROOT, first_difference
from verify_beam import digest, encoded

SCHEMA = "bopl-beam-scene-golden-v1"


def run_native(source, output):
    with source.open("rb") as inp, output.open("wb") as out:
        subprocess.run([str(ROOT / "native/target/release/dynamics-probe")], stdin=inp, stdout=out, check=True)


def compare(records):
    counts = Counter()
    failures = []
    for i, expected, actual in records:
        counts["cases"] += 1
        if isinstance(expected, list):
            counts["steps"] += len(expected)
            for state in expected:
                counts["callbacks"] += len(state["callbacks"])
                counts["active_hitbox_steps"] += sum(o["active"] and o["layer"] == 18 for o in state["scene"]["objects"])
                if state["graph"] is not None:
                    counts["rope_updates"] += sum(r["enabled"] for r in state["scene"]["rope_system"]["bodies"])
                counts["destroyed_pool_steps"] += sum(slot.get("destroyed", False) for slot in state["pool"])
                counts["composite_steps"] += bool(state["scene"]["composites"])
        else:
            counts["source_errors"] += 1
        if expected != actual:
            counts["mismatches"] += 1
            if len(failures) < 10:
                failures.append({"case": i, "difference": first_difference(expected, actual)})
    return {"match": not counts["mismatches"], **{key: counts[key] for key in
            ("cases", "steps", "callbacks", "active_hitbox_steps", "rope_updates", "composite_steps", "destroyed_pool_steps", "source_errors", "mismatches")}, "first_failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=917461)
    parser.add_argument("--reuse-unity", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--stress", action="store_true", help="36-step scenes with three beams per player")
    args = parser.parse_args()
    name = "beam-scene-stress" if args.stress else "beam-scene"
    fixture = ROOT / f"native/fixtures/{name}.jsonl.gz"
    directory = ROOT / f"native/artifacts/{name}"
    directory.mkdir(exist_ok=True, parents=True)
    source, output, native = [directory / name for name in ("input.jsonl", "unity.jsonl", "rust.jsonl")]
    if args.golden:
        with gzip.open(fixture, "rt") as stream:
            header = json.loads(next(stream)); records = [json.loads(line) for line in stream]
        if header["schema"] != SCHEMA or header["cases"] != len(records):
            raise ValueError("Invalid beam scene fixture")
        result = subprocess.run([str(ROOT / "native/target/release/dynamics-probe")],
                                input=encoded(r["query"] for r in records), text=True, capture_output=True, check=True)
        actual = [json.loads(line) for line in result.stdout.splitlines()]
        report = compare((i, record["expected"], value) for i, (record, value) in enumerate(zip(records, actual, strict=True)))
    else:
        if args.cases < 1: parser.error("--cases must be positive")
        raw = encoded(queries(args.cases, args.seed, stress=args.stress))
        if args.reuse_unity:
            if source.read_text() != raw: raise ValueError("Cannot reuse changed beam scene input")
        else:
            source.write_text(raw)
            env = dict(os.environ, BOPL_KERNEL_INPUT=str(source), BOPL_KERNEL_OUTPUT=str(output))
            for key in ("BOPL_TRACE", "BOPL_TRACE_TICKS", "BOPL_TRACE_BATCH"): env.pop(key, None)
            with (directory / "stdout.log").open("w") as out:
                subprocess.run([str(ROOT / "Builds/Linux/BoplBattle.x86_64"), str(ROOT / "native/fixtures/movement.rep"),
                                "-batchmode", "-nographics", "-logFile", str(directory / "unity.log")],
                               env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=600)
            if "REFERENCE_KERNEL_COMPLETE" not in (directory / "unity.log").read_text(): raise ValueError("Incomplete Unity beam scene run")
            (directory / "reference-sha256.json").write_text(json.dumps({"assembly": digest(ROOT / "Builds/Linux/BoplBattle_Data/Managed/Assembly-CSharp.dll"), "input": digest(source), "output": digest(output)}))
        hashes = json.loads((directory / "reference-sha256.json").read_text())
        if hashes["input"] != digest(source) or hashes["output"] != digest(output): raise ValueError("Beam scene reference provenance mismatch")
        run_native(source, native)
        with output.open() as left, native.open() as right:
            report = compare((i, json.loads(a), json.loads(b)) for i, (a, b) in enumerate(zip(left, right, strict=True)))
        if report["cases"] != args.cases: raise ValueError("Incomplete beam scene corpus")
        report["provenance"] = hashes
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["match"]:
            first, extra = (6, 14) if args.stress else (12, 36)
            selected = set(range(min(first, args.cases))) | set(random.Random(args.seed).sample(range(args.cases), min(extra, args.cases)))
            records = []
            with source.open() as inp, output.open() as out:
                for i, (q, e) in enumerate(zip(inp, out, strict=True)):
                    expected = json.loads(e)
                    if i in selected or isinstance(expected, dict): records.append({"query": json.loads(q), "expected": expected})
            header = {"schema": SCHEMA, "cases": len(records), "provenance": hashes}
            fixture.write_bytes(gzip.compress(encoded([header, *records]).encode(), mtime=0))
    print(json.dumps(report, indent=2))
    return int(not report["match"])


if __name__ == "__main__":
    raise SystemExit(main())
