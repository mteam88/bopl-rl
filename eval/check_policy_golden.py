"""Bit-identity regression of a frozen checkpoint against a recorded game.

Records (--record) or verifies the observations, raw decoder rows (logits +
value), recurrent state and sampled actions of a checkpoint playing itself on
maps 0-2 through the research environment. The fixture was recorded on the
layout-v3 code before policy v4 existed (commit f0131c9, sim
a73a3f25...): a v3 checkpoint must keep producing exactly these bytes through
every v4 change to the observation writer, the CPU evaluator and the loader.

    venv/bin/python eval/check_policy_golden.py                 # verify
    venv/bin/python eval/check_policy_golden.py --record        # re-record (only with a reason)
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

DEFAULT_CHECKPOINT = ROOT / 'native/artifacts/rl/sweeps/v3-base-007/000/checkpoints/policy-1333362688.bin'
DEFAULT_FIXTURE = HERE / 'fixtures/policy-golden-v3b-007.npz'
MAPS = (0, 1, 2)
SEED_BASE = 50001
DECISIONS = 300  # per map (1200 ticks = 20 s of play) or until the game ends


def play(checkpoint, decisions, layout_version=None):
    """Both players are the checkpoint; actions are sampled from a fixed generator."""
    from environment import Environment, configure
    import native_policy
    configure(Path(tempfile.mkdtemp(prefix='golden-')) / 'workers', ROOT / 'native/target/release/libbopl_sim.so')
    policy, meta = native_policy.load(checkpoint)
    out = dict(obs_sha=[], rows=[], actions=[], states=[], ticks=[], first_obs=[])
    with Environment(seed=SEED_BASE, limit=7200, delay=9) as env:
        if layout_version is not None:
            env.set_layout(0, layout_version); env.set_layout(1, layout_version)
        for map_id in MAPS:
            env.reset(SEED_BASE + map_id, map_id)
            env.bot(0, 0); env.bot(1, 0)
            generator = torch.Generator().manual_seed(1000 + map_id)
            state = policy.initial(2)
            for step in range(decisions):
                if env.ended:
                    break
                observation = env.obs[:, :policy.obs_size].copy()
                mask = env.mask.copy()
                with torch.no_grad():
                    rows, state = policy.step_raw(torch.from_numpy(observation), state)
                    logits, _ = policy.game_logits(rows)
                    logits = native_policy.masked(logits, mask)
                    actions = native_policy.sample(logits, generator)
                out['obs_sha'].append(hashlib.sha256(observation.tobytes()).hexdigest())
                out['rows'].append(rows.numpy().copy())
                out['states'].append(state.numpy().copy())
                out['actions'].append(actions.numpy().copy())
                out['ticks'].append(env.state.header.episode_ticks)
                if step < 4:
                    out['first_obs'].append(observation)
                env.step(actions.numpy().astype(np.float32))
    return dict(obs_sha=np.array(out['obs_sha']), rows=np.stack(out['rows']), states=np.stack(out['states']),
                actions=np.stack(out['actions']), ticks=np.array(out['ticks']), first_obs=np.stack(out['first_obs'])), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--decisions', type=int, default=DECISIONS)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--layout', type=int, default=None,
                        help='Force an observation layout version on both players (default: the checkpoint decides)')
    args = parser.parse_args()
    result, meta = play(args.checkpoint, args.decisions, args.layout)
    if args.record:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fixture, **result,
                            meta=json.dumps(dict(checkpoint=str(args.checkpoint.relative_to(ROOT)),
                                                 weights_sha256=meta['weights_sha256'], decisions=args.decisions,
                                                 maps=MAPS, seed_base=SEED_BASE)))
        print(f'recorded {len(result["obs_sha"])} decisions to {args.fixture}')
        return
    golden = np.load(args.fixture)
    info = json.loads(str(golden['meta']))
    if info['weights_sha256'] != meta['weights_sha256']:
        raise SystemExit('the fixture was recorded with a different checkpoint')
    failures = []
    for key in ('obs_sha', 'rows', 'states', 'actions', 'ticks'):
        got, want = result[key], golden[key]
        if got.shape != want.shape:
            failures.append(f'{key}: shape {got.shape} != {want.shape}')
        elif not np.array_equal(got, want):
            first = int(np.flatnonzero((got != want).reshape(got.shape[0], -1).any(1))[0])
            failures.append(f'{key}: first difference at decision {first} (tick {golden["ticks"][first]})')
    if failures:
        raise SystemExit('golden mismatch:\n  ' + '\n  '.join(failures))
    print(f'golden ok: {len(result["obs_sha"])} decisions bit-identical ({args.fixture.name})')


if __name__ == '__main__':
    main()
