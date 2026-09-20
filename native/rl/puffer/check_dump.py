#!/usr/bin/env python3
"""The trainer's checkpoint dump against its readers, on the trainer's own rows.

train.py --dump-rows runs one rollout at a tiny shape and saves, next to the
initial checkpoint it wrote from master_weights, the observations of 64
learner rows and the decoder rows (logits then value) the trainer's own
policy_forward produced for them from a zero recurrent state
(_C.bopl_policy_rows). This script loads that checkpoint the way the
evaluators do (native_policy.load: policy_eval.c, the CPU evaluator behind
evaluate.py, elo_ladder, the reference scorer and the in-game agent) and,
for policy v4, with the torch reference (research/policy_v4.py), and
asserts both reproduce the trainer's rows on the same observations. A reader
that walks the dump in another order than create_params registers it
(encoder, decoder, recurrent core) fails here rather than in a trial's
evaluations (the v4 MinGRU/decoder swap of 2026-09-16 was found by a trial).

Tolerance: the trainer runs TF32 GEMMs and bf16 set MLPs, the readers fp32;
the rows agree to ~1e-2 relative, a swapped or shifted reader is off by O(1).

    python native/rl/puffer/check_dump.py            # v3 (set) and v4, ~1 min, <2 GB GPU
    python native/rl/puffer/check_dump.py --only v4 --keep /tmp/dump
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'native/rl/research'))
import layout  # noqa: E402

PUFFERLIB = ROOT / 'native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9'
GAME = ROOT / 'native/target/release/libbopl_sim.so'
# A shape that constructs in seconds: four games, one buffer, the learner and
# one frozen bank, a short horizon; the rows come from the first rollout.
COMMON = ['--workers', '4', '--buffers', '1', '--banks', '2', '--horizon', '16', '--steps', '1',
          '--aim-fourier', '4', '--checkpoint-every', '1000000000', '--cudagraphs', '-1', '--no-eval',
          '--seed', '7', '--training-maps', '0,1,2', '--episode-ticks', '600']
CONFIGS = {
    'v3': ['--encoder', 'set', '--hidden', '128', '--layers', '1'],
    # v4 at 16 games: 24 learner rows x 16 steps for the rollout chain compare.
    'v4': ['--encoder', 'v4', '--hidden', '256', '--layers', '1', '--workers', '16'],
    # The production shape (sweep/configs/v4-lite-aim.toml): set inner 64, no FFN, critic 384, aim-only writer.
    'v4-lite': ['--encoder', 'v4', '--hidden', '256', '--layers', '1', '--workers', '16', '--v4-ablate', '10',
                '--v4-set-inner', '64', '--v4-ffn-dim', '0', '--v4-critic-dim', '384'],
}


def train_dump(name, out, game, chain_reference=None):
    """Run train.py --dump-rows at the tiny shape; the npz path."""
    run = out / name
    if run.exists():
        shutil.rmtree(run)
    dump = out / f'{name}.npz'
    env = dict(os.environ, BOPL_RL_BACKEND='rust', PYTHONPATH=str(PUFFERLIB))
    command = [sys.executable, str(HERE / 'train.py'), '--output', str(run), '--game', str(game),
               '--dump-rows', str(dump)] + COMMON + CONFIGS[name]
    if chain_reference is not None and name.startswith('v4'):
        command += ['--chain-reference', str(chain_reference / f'{name}.npz')]
    result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True)
    if result.returncode or not dump.exists():
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise RuntimeError(f'train.py --dump-rows failed for {name} ({result.returncode})')
    return dump


def readers(dump):
    """(name, step_raw) of every reader of the dump's checkpoint."""
    import torch
    import native_policy
    data = np.load(dump)
    checkpoint = Path(str(data['checkpoint']))
    meta = json.loads((checkpoint.parent / (checkpoint.name + '.json')).read_text())
    policy, _ = native_policy.load(checkpoint)
    out = [('cpu evaluator (policy_eval.c)', policy)]
    if meta['architecture'] == layout.V4_ARCHITECTURE:
        import policy_v4
        weights = torch.from_numpy(np.fromfile(checkpoint, dtype=np.float32).copy())
        shape = {k: meta['v4_shape'][k] for k in layout.V4_DEFAULTS}
        out.append(('torch reference (policy_v4.py)',
                    policy_v4.PolicyV4(weights, meta['hidden'], meta['num_layers'], meta.get('aim_fourier', 0),
                                       meta.get('aim_ar', 0), **shape)))
    return data, meta, out


def compare(dump, tolerance=2e-2, chain_reference=None):
    import torch
    data, meta, rs = readers(dump)
    obs = torch.from_numpy(data['observations'][:, :int(meta['obs_size'])].copy())
    expected = torch.from_numpy(data['rows'])
    report = dict(dump=str(dump), architecture=meta['architecture'], rows=int(obs.shape[0]),
                  width=int(expected.shape[1]), readers={})
    scale = max(1.0, expected[:, :-1].abs().max().item())
    for name, reader in rs:
        rows, _ = reader.step_raw(obs, torch.zeros(meta['num_layers'], obs.shape[0], meta['hidden']))
        if tuple(rows.shape) != tuple(expected.shape):
            raise AssertionError(f'{name}: rows {tuple(rows.shape)} vs trainer {tuple(expected.shape)}')
        logits = ((rows[:, :-1] - expected[:, :-1]).abs().max() / scale).item()
        value = (rows[:, -1] - expected[:, -1]).abs().max().item()
        report['readers'][name] = dict(logits_rel_max=logits, value_abs_max=value,
                                       pass_all=logits < tolerance and value < tolerance)
    # Packed observation transport (bopl_obs_pack.h): every unpacked device row of
    # the dump's rollout must equal its host row bit for bit (-1 = dense transport).
    transport = int(data['transport_mismatches']) if 'transport_mismatches' in data else -1
    report['transport_mismatches'] = transport
    # The batched rollout chain (bopl_rollout.cu, _C.bopl_chain_rows) on the
    # same rows: the trainer's own eager rows within the readers' tolerance
    # (mma.sync vs cuBLAS TF32 accumulate differently), and bit for bit against
    # a reference build's chain on the reference's observations and weights.
    if 'chain_rows' in data:
        chain = torch.from_numpy(data['chain_rows'])
        if tuple(chain.shape) != tuple(expected.shape):
            raise AssertionError(f'rollout chain: rows {tuple(chain.shape)} vs trainer {tuple(expected.shape)}')
        logits = ((chain[:, :-1] - expected[:, :-1]).abs().max() / scale).item()
        value = (chain[:, -1] - expected[:, -1]).abs().max().item()
        report['readers']['rollout chain (bopl_rollout.cu)'] = dict(logits_rel_max=logits, value_abs_max=value,
                                                                     pass_all=logits < tolerance and value < tolerance)
    if chain_reference is not None and 'chain_rows_reference' in data:
        ref = np.load(chain_reference)['chain_rows_all']
        got = data['chain_rows_reference']
        differing = int((ref != got).sum()) if ref.shape == got.shape else ref.size
        report['chain_reference'] = dict(values=int(ref.size), differing=differing,
                                         abs_max=float(np.abs(ref - got).max()) if ref.shape == got.shape else float('inf'))
    report['pass_all'] = (all(r['pass_all'] for r in report['readers'].values()) and transport <= 0
                          and report.get('chain_reference', {}).get('differing', 0) == 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', choices=sorted(CONFIGS), action='append')
    parser.add_argument('--game', type=Path, default=GAME)
    parser.add_argument('--keep', type=Path, help='directory for the runs and dumps (default: a temp dir)')
    parser.add_argument('--reuse', action='store_true', help='compare existing dumps under --keep without training')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--chain-reference', type=Path, metavar='DIR',
                        help='the --keep directory of another build\'s run: this build\'s rollout chain on each v4 '
                             'dump\'s observations and checkpoint must reproduce its chain_rows bit for bit')
    args = parser.parse_args()
    out = args.keep or Path(tempfile.mkdtemp(prefix='bopl-check-dump-'))
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.only or sorted(CONFIGS):
        dump = out / f'{name}.npz' if args.reuse else train_dump(name, out, args.game, args.chain_reference)
        report = compare(dump, chain_reference=args.chain_reference / f'{name}.npz' if args.chain_reference and name.startswith('v4') else None)
        reports.append(report)
        if not args.json:
            print(f"{name}: {report['rows']} rows x {report['width']} ({report['architecture']}), "
                  f"transport mismatches {report['transport_mismatches']}")
            for reader, r in report['readers'].items():
                print(f"  {reader:34s} logits rel max {r['logits_rel_max']:.2e}  value abs max {r['value_abs_max']:.2e}"
                      f"  {'ok' if r['pass_all'] else 'FAIL'}")
            if 'chain_reference' in report:
                c = report['chain_reference']
                print(f"  rollout chain vs reference build    {c['differing']} of {c['values']} values differ"
                      f"  (abs max {c['abs_max']:.2e})  {'ok' if c['differing'] == 0 else 'FAIL'}")
    if args.json:
        print(json.dumps(reports, indent=1))
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    if not all(r['pass_all'] for r in reports):
        sys.exit(1)


if __name__ == '__main__':
    main()
