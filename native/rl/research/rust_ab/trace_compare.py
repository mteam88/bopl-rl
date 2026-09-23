#!/usr/bin/env python3
"""Exact post-hoc comparator for the trace-only rust_ab probes.

Most probes in this directory assert internally and exit non-zero when the
backends disagree. A minority only *print* a per-tick trace for each backend
and always `return 0` -- for those the comparison lives here, and a suite that
checks exit status alone will report them as passing unconditionally. See
README.md "Trace-only probes".

Two rules this tool exists to enforce:

* **Exact equality, no tolerance.** Under `--features q32` both backends run
  the same Q32.32 integer arithmetic, so any difference at all is a real
  divergence. One raw unit is 1/2^32 ~= 2.33e-10; a tolerance as small as
  1e-6 hides the entire class of one-raw-unit bugs that this suite exists to
  catch (see the `Box.UpdateExtents` and sudden-death fixes).
* **Fail closed.** If no comparable content is found, that is an error (exit
  2), not a pass. A probe whose output format drifts must not silently become
  a no-op. Probes that genuinely assert internally are declared with
  `--self-asserting`, which turns "nothing to compare" into a clean exit.

Recognised formats, in order:

1. CSV whose header pairs `net_<field>` with `rust_<field>`.
2. CSV with a leading `backend` column: rows are grouped by their key columns
   (tick, map, angle, ... -- see KEY_COLUMNS) and every backend's row for a
   key must exist and agree. A key seen for one backend but not another is a
   divergence, not a skip.
3. Free-form lines carrying `net=(...) rust=(...)`, `| net ... | rust ...`,
   or `net_<field>=V ... rust_<field>=V`.

Exit status: 0 agreement, 1 divergence, 2 nothing comparable / bad input.
"""
from __future__ import annotations

import argparse
import re
import sys

# Column names that identify *which* sample a row is, rather than the measured
# values being compared. Matched case-insensitively against the whole name.
KEY_COLUMNS = {
    "backend", "map", "angle", "seed", "episode", "decision", "tick",
    "release_tick", "level_tick", "world_tick", "frame", "step", "pass",
    "case", "scenario", "player", "slot", "entity", "index", "id", "slime",
}

PAIR_RE = re.compile(r"net=\(([^)]*)\)\s*rust=\(([^)]*)\)")
BARS_RE = re.compile(r"\|\s*net\s+(.*?)\s*\|\s*rust\s+(.*?)\s*$")
KV_RE = re.compile(r"\b(net|rust)_(\w+)=(\S+)")


class Result:
    def __init__(self) -> None:
        self.comparisons = 0
        self.problems: list[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)


def compare_prefixed(lines: list[str], out: Result) -> bool:
    header = lines[0].split(",")
    net = {c[4:]: i for i, c in enumerate(header) if c.startswith("net_")}
    rust = {c[5:]: i for i, c in enumerate(header) if c.startswith("rust_")}
    fields = sorted(set(net) & set(rust))
    if not fields:
        return False
    for number, line in enumerate(lines[1:], start=2):
        if line == lines[0]:
            continue  # a concatenated run repeats the header per invocation
        row = line.split(",")
        if len(row) != len(header):
            continue
        for field in fields:
            out.comparisons += 1
            if row[net[field]] != row[rust[field]]:
                out.fail(
                    f"line {number} field {field}: "
                    f"net {row[net[field]]} != rust {row[rust[field]]}"
                )
    return True


def compare_backend_rows(lines: list[str], out: Result) -> bool:
    header = lines[0].split(",")
    if header[0] != "backend":
        return False
    keys = [i for i, c in enumerate(header) if c.lower() in KEY_COLUMNS and i != 0]
    values = [i for i in range(1, len(header)) if i not in keys]
    if not values:
        return False
    # Preserve first-seen order so the reported divergence is the earliest one.
    groups: dict[tuple[str, ...], dict[str, tuple[int, list[str]]]] = {}
    order: list[tuple[str, ...]] = []
    backends: list[str] = []
    for number, line in enumerate(lines[1:], start=2):
        if line == lines[0]:
            continue  # a concatenated run repeats the header per invocation
        row = line.split(",")
        if len(row) != len(header):
            continue
        if row[0] not in backends:
            backends.append(row[0])
        key = tuple(row[i] for i in keys)
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][row[0]] = (number, row)
    if len(backends) < 2:
        return False
    for key in order:
        rows = groups[key]
        located = ",".join(
            f"{header[i]}={value}" for i, value in zip(keys, key)
        )
        missing = [b for b in backends if b not in rows]
        if missing:
            # One backend emitted a sample the other never did. That is a
            # divergence in its own right -- an entity that exists on one side
            # and not the other -- and must not be skipped as "unpaired".
            out.comparisons += 1
            out.fail(f"{located}: no row from {', '.join(missing)}")
            continue
        base_backend = backends[0]
        _, base = rows[base_backend]
        for other_backend in backends[1:]:
            number, other = rows[other_backend]
            for i in values:
                out.comparisons += 1
                if base[i] != other[i]:
                    out.fail(
                        f"line {number} {located} field {header[i]}: "
                        f"{base_backend} {base[i]} != {other_backend} {other[i]}"
                    )
    return True


def compare_freeform(lines: list[str], out: Result) -> bool:
    found = False
    for number, line in enumerate(lines, start=1):
        for a, b in PAIR_RE.findall(line):
            found = True
            out.comparisons += 1
            if a != b:
                out.fail(f"line {number}: net=({a}) rust=({b})")
        match = BARS_RE.search(line)
        if match:
            found = True
            out.comparisons += 1
            if match.group(1) != match.group(2):
                out.fail(
                    f"line {number}: net {match.group(1)} != rust {match.group(2)}"
                )
        pairs = KV_RE.findall(line)
        if pairs:
            found = True
            fields: dict[str, dict[str, str]] = {}
            for side, name, value in pairs:
                fields.setdefault(name, {})[side] = value
            for name, sides in sorted(fields.items()):
                if "net" in sides and "rust" in sides:
                    out.comparisons += 1
                    if sides["net"] != sides["rust"]:
                        out.fail(
                            f"line {number} field {name}: "
                            f"net {sides['net']} != rust {sides['rust']}"
                        )
    return found


def compare(path: str) -> Result:
    out = Result()
    try:
        with open(path, errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        out.fail(f"cannot read: {error}")
        return out
    lines = [line for line in lines if line.strip()]
    if not lines:
        return out
    if "," in lines[0]:
        if compare_prefixed(lines, out):
            return out
        if compare_backend_rows(lines, out):
            return out
    compare_freeform(lines, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="probe stdout captures")
    parser.add_argument(
        "--self-asserting",
        action="store_true",
        help="the probe compares internally; finding nothing here is fine",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only report divergences"
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=3,
        help="divergences to print per file (default 3)",
    )
    arguments = parser.parse_args()

    status = 0
    for path in arguments.paths:
        name = path.rsplit("/", 1)[-1]
        out = compare(path)
        if out.problems:
            status = 1
            print(f"TRACE-DIFF {name}: {len(out.problems)} divergence(s)")
            for problem in out.problems[: arguments.max_report]:
                print(f"  {problem}")
            remaining = len(out.problems) - arguments.max_report
            if remaining > 0:
                print(f"  ... and {remaining} more")
        elif out.comparisons == 0:
            if arguments.self_asserting:
                if not arguments.quiet:
                    print(f"TRACE-SELF {name}: compares internally")
            else:
                status = max(status, 2)
                print(
                    f"TRACE-NONE {name}: no comparable output -- the probe "
                    f"prints nothing this tool can pair, so it cannot fail. "
                    f"Declare --self-asserting or fix the trace format."
                )
        elif not arguments.quiet:
            print(f"TRACE-OK   {name}: {out.comparisons} comparisons")
    return status


if __name__ == "__main__":
    sys.exit(main())
