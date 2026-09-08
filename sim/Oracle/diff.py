#!/usr/bin/env python3
"""Diff a Unity oracle trace against a port trace.

    python sim/Oracle/diff.py <unity.jsonl.gz> <port.jsonl.gz> [--ignore-presentation] [--summary]

Thin wrapper over native/tools/reference.py (read_trace, first_difference).
Exit codes: 0 full match, 1 divergence, 2 header (setup/schema) mismatch.

Fields listed in native/data/process-identity-fields.json are ALWAYS ignored:
they hold raw GetInstanceID() integers, which are process-global in Unity and
a monotonic shim counter in the port, so they cannot match by construction.
--ignore-presentation additionally applies the audited cosmetic exclusions
from native/data/presentation-fields.json (the projection verify_replay.py
uses).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native/tools"))
from reference import read_trace, first_difference  # noqa: E402


def load_fields(name):
    return frozenset(json.loads((ROOT / "native/data" / name).read_text())["fields"])


# "$type" values are .NET type names; for generic types they embed
# assembly-qualified arguments (RingBuffer`1[[MonoUpdatable, Assembly-CSharp,
# Version=0.0.0.0, ...]]). The assembly identity is build metadata - Unity says
# Assembly-CSharp/mscorlib, the port says Gameplay/System.Private.CoreLib - so
# strip ", assembly, Version=..." from every bracketed type argument before
# comparing. Applied repeatedly for nested generics.
_ASSEMBLY_QUALIFIER = re.compile(r"\[([^\[\],]+), [^\[\]]+\]")


def normalize_types(node):
    if isinstance(node, dict):
        return {k: normalize_types(v) for k, v in node.items()}
    if isinstance(node, list):
        return [normalize_types(v) for v in node]
    if isinstance(node, str) and "[[" in node:
        previous = None
        while previous != node:
            previous, node = node, _ASSEMBLY_QUALIFIER.sub(r"[\1]", node)
    return node


def summarize(left, right, ignored):
    """Which top-level sections of a trace row differ (players, updatables, ...)."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return ["<row kind/shape>"]
    lv, rv = left.get("value"), right.get("value")
    if not isinstance(lv, dict) or not isinstance(rv, dict):
        return sorted(set(left) ^ set(right)) or ["<value shape>"]
    sections = []
    for key in sorted(set(lv) | set(rv)):
        if first_difference(lv.get(key), rv.get(key), ignored_fields=ignored):
            sections.append(key)
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unity", type=Path)
    parser.add_argument("port", type=Path)
    parser.add_argument("--ignore-presentation", action="store_true",
                        help="Also exclude the audited cosmetic fields (presentation-fields.json)")
    parser.add_argument("--summary", action="store_true",
                        help="List which top-level sections differ at the first diverging row")
    parser.add_argument("--exclude", action="append", default=[], metavar="FIELD",
                        help="Additional field name to ignore (repeatable). Used for "
                             "known export gaps, e.g. Player.playerDied/playerWon whose "
                             "subscribers come from the unexported menu flow.")
    args = parser.parse_args()

    ignored = load_fields("process-identity-fields.json") | frozenset(args.exclude)
    if args.ignore_presentation:
        ignored |= load_fields("presentation-fields.json")

    left, right = read_trace(args.unity), read_trace(args.port)
    header_l, header_r = next(left), next(right)
    for key in ("schema", "setup"):
        if header_l.get(key) != header_r.get(key):
            print(f"header mismatch: {key}")
            print(f"  left:  {json.dumps(header_l.get(key))}")
            print(f"  right: {json.dumps(header_r.get(key))}")
            return 2

    ticks = 0
    while True:
        a, b = next(left, None), next(right, None)
        if a is not None:
            a = normalize_types(a)
        if b is not None:
            b = normalize_types(b)
        if a and b and a.get("kind") == b.get("kind") == "end":
            a, b = dict(a), dict(b)
            a.pop("reason", None)  # transport-specific (tick-limit vs replay end)
            b.pop("reason", None)
        difference = first_difference(a, b, ignored_fields=ignored)
        if difference:
            path, lval, rval = difference
            print(f"matched ticks: {ticks}")
            print(f"first difference at: {path or '<row>'}")
            print(f"  left:  {json.dumps(lval, default=str)[:500]}")
            print(f"  right: {json.dumps(rval, default=str)[:500]}")
            if args.summary:
                print("differing sections:", ", ".join(summarize(a, b, ignored)) or "<none at top level>")
            return 1
        if a is None:
            break
        if a["kind"] == "state":
            ticks += 1
    print(f"match: {ticks} ticks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
