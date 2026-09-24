#!/usr/bin/env python
"""Prove that Environment.step_ticks (one game tick at a time, for 60 frames/s
play) plays the same game as Environment.step (bopl_rl_step_sampled(3, 1)).

Two fixed-seed games (scripted-vs-scripted, checkpoint-vs-scripted; two seeds
on two maps each by default) are played twice each, once per path, and compared on
everything the decision loop can observe: the .rep bytes (BOPLREP2: the
per-tick inputs of both players), every decision's observation, mask and
telemetry, the full state dump at every decision boundary (players,
entities, slimes), the death ticks and the final tick. A scripted bot decides
from the true state inside the environment on every decision, so any state
divergence would change its inputs and the replay; the checkpoint samples from
observations with a fixed torch generator, so any observation divergence would
change its actions.

    BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python eval/check_tick_step.py \
        --checkpoint .../checkpoints/policy-<step>.bin [--seed N ...] [--map M ...]

Prints one JSON line per game and TICK_STEP_OK when every comparison holds.
"""
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from environment import Environment, configure
from evaluation_protocol import scripted_opponent
import layout, native_policy
from native_policy import decide

ROOT = HERE.parent
RUST_SIM = ROOT / 'native/target/release/libbopl_sim.so'


def play(env, seed, map_id, bots, policy, per_tick, out):
    """One game on a fresh reset; returns everything comparable."""
    env.reset(seed, map_id)
    env.bot(0, 0); env.bot(1, 0)
    for player, name in bots.items():
        env.bot(player, *scripted_opponent(name))
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    state = policy.initial(2, 'cpu') if policy is not None else None
    log = dict(obs=[], mask=[], telemetry=[], frames=[], tick_frames=[], deaths=[], ticks_per_decision=[])
    deaths = [0, 0]
    t0 = time.perf_counter()
    while not env.ended:
        observation = torch.from_numpy(env.obs.copy())
        mask = env.mask.copy()
        log['obs'].append(observation.numpy().copy())
        log['mask'].append(mask)
        actions = torch.zeros(2, layout.NUM_ATNS, dtype=torch.int64)
        if policy is not None:
            with torch.no_grad():
                _, _, actions, state = decide(policy, observation, mask, state, generator)
        if per_tick:
            n = 0
            for _ in env.step_ticks(actions.numpy()):
                n += 1
                frame = env.frame()
                log['tick_frames'].append(frame)
            log['ticks_per_decision'].append(n)
        else:
            env.step(actions.numpy())
        frame = env.frame()
        log['frames'].append(frame)
        log['telemetry'].append(env.telemetry.copy())
        for i, p in enumerate(frame['players']):
            if p['deaths'] != deaths[i]:
                deaths[i] = p['deaths']; log['deaths'].append((i, frame['t']))
    log['seconds'] = time.perf_counter() - t0
    replay = Path(out) / ('tick' if per_tick else 'step') / f'{seed}-{map_id}.rep'
    replay.parent.mkdir(parents=True, exist_ok=True)
    env.replay(replay)
    log['rep'] = replay.read_bytes()
    log['final'] = env.frame()
    return log


def compare(a, b):
    """Differences between the step path (a) and the per-tick path (b)."""
    problems = []
    if a['rep'] != b['rep']:
        problems.append('replay bytes differ')
    if len(a['obs']) != len(b['obs']):
        problems.append(f"decision count {len(a['obs'])} vs {len(b['obs'])}")
    for name in ('obs', 'mask', 'telemetry'):
        for d, (x, y) in enumerate(zip(a[name], b[name])):
            if not np.array_equal(x, y):
                problems.append(f'{name} differs at decision {d}'); break
    for d, (x, y) in enumerate(zip(a['frames'], b['frames'])):
        if x != y:
            problems.append(f'decision-boundary state differs at decision {d} (tick {x["t"]})'); break
    # The last per-tick frame of each decision is the step path's frame (the final decision may be short).
    boundary, at = [], 0
    for n in b['ticks_per_decision']:
        at += n; boundary.append(b['tick_frames'][at - 1])
    if len(boundary) != len(a['frames']) or any(x != y for x, y in zip(a['frames'], boundary)):
        problems.append('per-tick frames at decision boundaries differ from the step frames')
    ticks = [f['t'] for f in b['tick_frames']]
    if ticks != list(range(ticks[0], ticks[0] + len(ticks))) or (ticks and ticks[0] != 1):
        problems.append('per-tick frames are not consecutive ticks from 1')
    if a['deaths'] != b['deaths']:
        problems.append(f"death ticks {a['deaths']} vs {b['deaths']}")
    if a['final'] != b['final']:
        problems.append('final state differs')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scripted', default='mixed-hard')
    parser.add_argument('--scripted-b', default='archer-hard', help='Opponent for the scripted-vs-scripted game')
    parser.add_argument('--seed', type=int, action='append', default=None, help='Fixed seed (repeatable; default 4242 and 90001)')
    parser.add_argument('--map', type=int, action='append', default=None, help='Map (repeatable; default 0 and 3)')
    parser.add_argument('--game', type=Path, default=RUST_SIM)
    parser.add_argument('--out', type=Path, default=None, help='Where the .rep files go (default: a temp dir)')
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy, meta = native_policy.load(args.checkpoint)
    config = meta.get('config', {})
    limit, delay = config.get('episode_ticks', 7200), config.get('observation_delay_ticks', 9)
    out = args.out or Path(tempfile.mkdtemp(prefix='bopl-tick-step-'))
    configure(out / 'workers', args.game)
    ok = True
    games = [dict(name='scripted-vs-scripted', bots={0: args.scripted, 1: args.scripted_b}, policy=None, limit=7200, delay=9),
             dict(name='checkpoint-vs-scripted', bots={0: args.scripted}, policy=policy, limit=limit, delay=delay)]
    for game, seed, map_id in ((g, s, m) for g in games for s in (args.seed or [4242, 90001]) for m in (args.map or [0, 3])):
        env = Environment(seed, map_id, game['limit'], game['delay'])
        try:
            a = play(env, seed, map_id, game['bots'], game['policy'], False, out)
            b = play(env, seed, map_id, game['bots'], game['policy'], True, out)
        finally:
            env.close()
        problems = compare(a, b)
        ok &= not problems
        print(json.dumps(dict(
            game=game['name'], seed=seed, map=map_id, bots=game['bots'], checkpoint=str(args.checkpoint) if game['policy'] else None,
            decisions=len(a['obs']), ticks=a['final']['t'], per_tick_frames=len(b['tick_frames']),
            rep_sha256=hashlib.sha256(a['rep']).hexdigest()[:16], rep_bytes=len(a['rep']),
            deaths=a['deaths'], final_kills=[p['kills'] for p in a['final']['players']],
            step_seconds=round(a['seconds'], 3), tick_seconds=round(b['seconds'], 3),
            identical=not problems, problems=problems)), flush=True)
    print('TICK_STEP_OK' if ok else 'TICK_STEP_FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
