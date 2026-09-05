"""Differential test Rust against the original C# arithmetic, including failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OPS = "add sub mul div slow_add slow_sub slow_mul rem slow_rem neg abs slow_abs floor ceil round sqrt sin cos fast_sin fast_cos tan acos atan atan2".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=67484)
    parser.add_argument("--report", type=Path, default=ROOT / "native/artifacts/math-report.json")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "native/data/reference.json").read_text())
    source_hash = hashlib.sha256((ROOT / manifest["math_source"]).read_bytes()).hexdigest()
    if source_hash != manifest["math_source_sha256"]:
        raise ValueError("math oracle source changed from pinned reference")
    for filename, entry in manifest["tables"].items():
        if hashlib.sha256((ROOT / "native/data" / filename).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"lookup table changed: {filename}")
    rng = random.Random(args.seed)
    boundaries = {0, -1, 1, -(1 << 63), (1 << 63) - 1}
    for n in (1 << 31, 1 << 32, 6746518852, 13493037704, 26986075409, 1 << 48, 1 << 62):
        for sign in (-1, 1):
            for delta in (-1, 0, 1):
                boundaries.add(sign * n + delta)
    pairs = list(itertools.product(sorted(boundaries), repeat=2))
    for i in range(args.random):
        bits = 64 if i % 2 else 43
        pairs.append((rng.randrange(-(1 << (bits-1)), 1 << (bits-1)), rng.randrange(-(1 << (bits-1)), 1 << (bits-1))))
    cases = [(op, a, b) for op in OPS for a, b in pairs]
    # Exercise valid inverse-cosine inputs densely, not just rejected full-range values.
    cases += [("acos", rng.randrange(-(1 << 32), (1 << 32) + 1), 0) for _ in range(args.random)]
    data = "".join(f"{op}\t{a}\t{b}\n" for op, a, b in cases).encode()
    commands = [
        ["dotnet", str(ROOT / "native/tools/math-oracle/bin/Release/net8.0/math-oracle.dll")],
        [str(ROOT / "native/target/release/math-probe")],
    ]
    def run(command):
        return subprocess.run(command, input=data, capture_output=True, check=True).stdout.decode().splitlines()
    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(run, commands))
    if len(left) != len(cases) or len(right) != len(cases):
        raise ValueError(f"incomplete results: {len(cases)} cases, {len(left)} C#, {len(right)} Rust")
    failures = []
    counts = {op: {"cases": 0, "mismatches": 0, "expected_errors": 0} for op in OPS}
    for case, a, b in zip(cases, left, right):
        counts[case[0]]["cases"] += 1
        counts[case[0]]["expected_errors"] += not a.lstrip("-").isdigit()
        if a != b:
            counts[case[0]]["mismatches"] += 1
            if len(failures) < 20:
                failures.append({"operation": case[0], "a": case[1], "b": case[2], "csharp": a, "rust": b})
    report = {"seed": args.seed, "cases": len(cases), "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "reference": manifest["reference_commit"], "source_sha256": source_hash,
        "operations": counts, "first_failures": failures, "match": not failures}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"match": not failures, "cases": len(cases), "mismatches": sum(v["mismatches"] for v in counts.values()), "first_failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
