"""Compare two short training runs for the bit-off gate. The trainer is not bit-deterministic at a working learning
rate, but two runs agree byte for byte where it counts:
  metrics <a> <b>   runs at --learning-rate 1e-30: the weights never move, so every metrics.jsonl record (losses,
                    gradient norm, KL, behaviour, match results) depends only on the rows. The records must be equal
                    apart from wall-clock keys, and every checkpoint must hash the same.
  rows <a> <b>      runs with --dump-rows: the trainer's own GPU rows over the first rollout (chain_observations,
                    chain_rows_all) and the transport check's mismatch count must be equal, and the mismatches 0.
  zero <a>          one --dump-rows run: transport mismatches (GPU rows vs the CPU writer) must be 0.
Prints one line, exits 1 on a difference.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

CLOCK = {'elapsed', 'sps', 'learner_sps', 'rollout_seconds', 'train_seconds', 'wall_seconds', 'native_perf', 'seconds'}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in CLOCK and not k.endswith('_seconds') and not k.endswith('_sps')}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def metrics(a, b):
    ra = [strip(json.loads(line)) for line in open(a / 'metrics.jsonl')]
    rb = [strip(json.loads(line)) for line in open(b / 'metrics.jsonl')]
    if len(ra) != len(rb) or not ra:
        return False, f'{len(ra)} vs {len(rb)} records'
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            return False, f'record {i} differs in {keys[:8]}'
    ck = lambda d: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((d / 'checkpoints').glob('policy-*.bin'))}
    ca, cb = ck(a), ck(b)
    if ca != cb or not ca:
        return False, f'checkpoints differ ({len(ca)} vs {len(cb)})'
    last = ra[-1]
    return True, (f'{len(ra)} metrics records equal (last: update {last.get("update")}, step {last.get("step")}), '
                  f'{len(ca)} checkpoints equal (final {list(ca)[-1]} {list(ca.values())[-1][:16]})')


def rows_hash(d):
    z = np.load(d / 'rows.npz')
    return {k: hashlib.sha256(z[k].tobytes()).hexdigest()[:16] for k in ('chain_observations', 'chain_rows_all')}, int(z['transport_mismatches'])


def main():
    mode, a = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == 'metrics':
        ok, msg = metrics(a, pathlib.Path(sys.argv[3]))
    elif mode == 'rows':
        (ha, ma), (hb, mb) = rows_hash(a), rows_hash(pathlib.Path(sys.argv[3]))
        ok = ha == hb and ma == mb == 0
        msg = f'trainer rows {ha} vs {hb}; transport mismatches {ma} / {mb}'
    elif mode == 'zero':
        h, m = rows_hash(a)
        ok, msg = m == 0, f'transport mismatches {m} (rows {h})'
    else:
        raise SystemExit(__doc__)
    print(('PASS ' if ok else 'FAIL ') + msg)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
