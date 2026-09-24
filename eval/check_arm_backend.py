"""Prove which backend actually played each cell of a checkpoint-transfer arm.

A manifest records the library path a run was ASKED to use. That is not evidence:
the whole point of the exercise is that the evaluation stack picks its backend at
runtime, so a mistyped path, a stale environment variable or a silent fallback
would produce a full set of plausible-looking results under the wrong simulator.

The backends leave different fingerprints. evaluate.py points BOPL_RL_LOG_DIR at
<cell>/workers, and the C# oracle treats that as its -logFile: every oracle process
writes the three boot lines sim/RL.md calls expected (Steam unavailable, the
unexported Replay scene, a ParticleSystem.randomSeed overflow) plus its scene
export warnings. The Rust sim writes no such file at all. So a non-empty worker log
carrying SteamManager.Awake distinguishes them without trusting any metadata.

    python check_arm_backend.py --out DIR --arm oracle --expect oracle
    python check_arm_backend.py --out DIR --arm main   --expect rust

Note the asymmetry, and do not read more into it than it carries: `--expect
oracle` is positive evidence (the C# game's own output is present), while
`--expect rust` is only the absence of that evidence, which a cell whose worker
logs were never written or were not copied would also satisfy. Establish a Rust
arm's provenance by other means -- `checkpoint_transfer.py crosscheck` against a
cell you ran yourself and checked here is the cheap one.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_transfer import KITS, final_checkpoint

# Lines only the .NET/Unity game code can emit (sim/RL.md "three expected [error] lines").
ORACLE_MARKS = ('SteamManager.Awake', 'Steam unavailable in sim', 'Export[scene')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--expect', choices=('oracle', 'rust'), required=True)
    p.add_argument('--kits', nargs='*', default=KITS)
    a = p.parse_args()

    rows, bad = [], 0
    for kit in a.kits:
        _, step = final_checkpoint(kit)
        cell = a.out.resolve() / 'runs' / kit / a.arm / 'evaluations' / f'{step:09d}'
        if not (cell / 'result.json').is_file():
            rows.append(dict(kit=kit, state='missing'))
            continue
        logs = sorted((cell / 'workers').glob('*.log')) if (cell / 'workers').is_dir() else []
        hits = 0
        for log in logs:
            text = log.read_text(errors='replace')
            hits += sum(mark in text for mark in ORACLE_MARKS)
        seen = 'oracle' if hits else 'rust'
        ok = seen == a.expect
        bad += not ok
        rows.append(dict(kit=kit, state='ok' if ok else 'WRONG BACKEND', worker_logs=len(logs),
                         oracle_markers=hits, seen=seen))
        print(f"{kit:28s} worker_logs={len(logs):2d} oracle_markers={hits:2d} "
              f"-> {seen:6s} {'OK' if ok else 'MISMATCH, expected ' + a.expect}")

    played = [r for r in rows if r.get('state') != 'missing']
    (a.out.resolve() / f'backend-check-{a.arm}.json').write_text(json.dumps(rows, indent=1) + '\n')
    if not played:
        raise SystemExit('no completed cells were inspected: this check proves nothing')
    if bad:
        raise SystemExit(f'{bad} cell(s) ran on the wrong backend')
    print(f'\nOK: all {len(played)} completed {a.arm} cells ran on the {a.expect} backend')


if __name__ == '__main__':
    main()
