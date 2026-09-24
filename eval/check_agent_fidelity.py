"""Fidelity of the in-game agent shim (bopl_agent.h) against the research environment.

Two checks:

  sim    Drive a research Environment (eval_binding.c, any backend) decision by
         decision; hand every tick's state to the shim as the game would. At
         every decision boundary the shim's observation row, action mask and
         (policies) decoder output must be bit-identical to the environment's,
         and the shim's per-tick InputPacket must equal what the environment
         applied (recovered from the own-action tail of the next observation).
         For bots the two decide independently from the same states and seed,
         so a diverging decision would surface as a diverging tail.

  replay Take a .rep the real game auto-saved from a match against the shim
         (Tools/play-vs-agent.sh), replay the HUMAN side into the research
         environment on the Unity build with the shim on the other side, and
         compare the agent's InputPackets tick for tick with the ones recorded
         in the .rep. The shim is seeded with the round seed recorded in the
         result JSON (games.jsonl), so the only source of divergence would be
         the game itself.

Usage:
  check_agent_fidelity.py sim <checkpoint.bin | bot-name> [--seed N] [--maps 0,3,7] [--game LIB]
  check_agent_fidelity.py replay <match.rep> <checkpoint.bin | bot-name> --seed N --agent-index I [--game BUILD]
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import layout

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def sample_masked(rng, mask):
    """A random legal action (the "human" of the sim check): one index per head."""
    out = []
    lo = 0
    for size in layout.ACTION_SIZES:
        allowed = np.flatnonzero(mask[lo:lo+size])
        out.append(int(rng.choice(allowed)) if allowed.size else 0)
        lo += size
    return out

def check_sim(args):
    from environment import Environment, configure
    from agent_shim import Agent
    configure(Path('/tmp/play-unity/fidelity-logs'), args.game)
    spec = str(args.agent)
    maps = [int(m) for m in args.maps.split(',')]
    total_decisions = total_ticks = 0
    report = {'check': 'sim', 'agent': spec, 'rounds': []}
    with Agent(spec, players=2, player=1) as agent:
        report['agent_id'] = agent.id
        limit = agent.max_episode_ticks
        for round_index, map_id in enumerate(maps):
            seed = args.seed + round_index
            with Environment(seed=seed, map_id=map_id, limit=limit, delay=9) as env:
                # The research environment draws the episode seed from its stream: init() scrambles
                # env->rng once (LCG), then bopl_draw_config takes bopl_random (xorshift32). The shim
                # takes the episode seed directly (the game's seed), so replicate that draw here.
                x = (seed * 747796405 + 2891336453) & 0xffffffff
                x ^= (x << 13) & 0xffffffff; x ^= x >> 17; x ^= (x << 5) & 0xffffffff
                episode_seed = x
                if not agent.is_policy:
                    kind, level = bot_kind(spec)
                    env.bot(1, kind, level)
                agent.reset(episode_seed)
                rng = np.random.default_rng(seed)
                status, packet = agent.tick(env.state)      # decision 0 from S0
                decisions = 0; ticks = 0; mismatches = []
                # The environment writes the shim's side in the checkpoint's layout (v3 or v4,
                # layout.py), like evaluate.py; set_layout re-encodes the reset row so the
                # comparison below starts at the same S0 row the shim primes from.
                policy = None
                if agent.is_policy:
                    import native_policy
                    policy, _ = native_policy.load(spec)
                    env.set_layout(1, layout.layout_of(policy.architecture), ablate=native_policy.writer_ablation(policy))
                # Layout v4's tail (V4_PRIVILEGED floats from V4_PRIVILEGED_OFFSET) is the
                # critic's privileged block, which encodes the opponent's applied action:
                # the shim never sees the human's inputs and the actor never reads it
                # (bopl_v4_eval.h feeds it to the value head only), so compare the actor's
                # input, the row up to that offset; the logits check covers the rest.
                width = layout.V4_PRIVILEGED_OFFSET if env.layout_of(1) == layout.V4_LAYOUT_VERSION else layout.OBS_SIZE
                row = lambda: env.obs_row(1)[:width]
                if agent.observation.shape[0] != env.obs_row(1).shape[0]:
                    raise RuntimeError(f'shim row {agent.observation.shape} vs environment row {env.obs_row(1).shape} for {spec}')
                actor = lambda: agent.observation[:width]
                obs_ok = np.array_equal(actor(), row()); mask_ok = np.array_equal(agent.mask, env.mask[1])
                if not (obs_ok and mask_ok): mismatches.append(('reset', obs_ok, mask_ok))
                logits_checked = 0
                if policy is not None:
                    import torch
                    state = policy.initial(1)
                while not env.ended and ticks < limit + 8:
                    if policy is not None:
                        rows, state = policy.step_raw(torch.from_numpy(env.obs_row(1)[None].copy()), state)
                        # The shim's logits are the game logits of the button combo it sampled (aim_ar).
                        logits, value = policy.game_logits(rows, np.array([agent.decision[1]], dtype=np.int32))
                        if not np.array_equal(logits.numpy()[0], agent.logits[:layout.LOGITS]): mismatches.append(('logits', decisions))
                        logits_checked += 1
                    human = sample_masked(rng, env.mask[0])
                    decision = list(agent.decision)
                    applied = []
                    for left in env.step_ticks([human, decision]):
                        ticks += 1
                        applied.append(packet)
                        status, packet = agent.tick(env.state)
                        if status == 1: break
                    decisions += 1
                    # every tick of the finished decision carried the same latched packet; the environment's
                    # own-action tail (bopl_previous_action) encodes the action it actually applied.
                    if len(set(applied)) != 1: mismatches.append(('packet-varies', decisions, applied))
                    if not env.ended:
                        obs_ok = np.array_equal(actor(), row()); mask_ok = np.array_equal(agent.mask, env.mask[1])
                        if not (obs_ok and mask_ok): mismatches.append(('decision', decisions, obs_ok, mask_ok))
                        if len(mismatches) > 5: break
                total_decisions += decisions; total_ticks += ticks
                report['rounds'].append({'seed': seed, 'episode_seed': episode_seed, 'map': map_id, 'ticks': ticks, 'decisions': decisions,
                                         'logits_checked': logits_checked, 'ended': env.ended, 'terminal_status': status,
                                         'mismatches': [str(m) for m in mismatches]})
                print(f"map {map_id} seed {seed}: {ticks} ticks, {decisions} decisions, "
                      f"{logits_checked} logits compared, ended={env.ended}, mismatches={len(mismatches)}", flush=True)
    report['total_ticks'] = total_ticks; report['total_decisions'] = total_decisions
    report['ok'] = all(not r['mismatches'] and r['ended'] for r in report['rounds'])
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1

def bot_kind(name):
    from evaluation_protocol import scripted_opponent
    return scripted_opponent(name)

def check_replay(args):
    from replay_data import decode, ReplayWorld
    from agent_shim import Agent
    rep = Path(args.replay)
    meta, inputs = decode(rep)
    players = meta['players']
    agent_index = args.agent_index
    spec = str(args.agent)
    with Agent(spec, players=len(players), player=agent_index) as agent:
        warmup = agent.warmup_ticks
        if players[agent_index]['abilities'] != agent.abilities:
            raise SystemExit(f"replay player {agent_index} carries abilities {players[agent_index]['abilities']}, the agent trained with {agent.abilities}")
        # The recording's first `warmup` frames must be the frozen warmup (RlAgent feeds zeros to everyone).
        frozen = inputs[:warmup]
        if np.any(frozen['angle']) or np.any(frozen['buttons']):
            raise SystemExit('replay does not start with the neutral training warmup; was it recorded with BOPL_AGENT?')
        agent.reset(args.seed)
        agent_id = agent.id
        # The research world runs the warmup inside its reset (config.warmup_ticks), so its
        # episode tick 0 is the recording's frame `warmup`; frame f holds the inputs applied
        # to the tick that takes S_f to S_{f+1}. The game never truncates (RlAgent plays past
        # max_episode_ticks), so the world's limit is the recording's length.
        log = Path(os.environ.get('BOPL_FIDELITY_LOG', '/tmp/bopl-agent-fidelity-replay.log'))
        world = ReplayWorld(meta, log, limit=meta['ticks'] + 1, game=args.game, warmup_ticks=warmup)
        compared = matched = 0; first_mismatch = None; ended = False; mismatches = []
        try:
            for f in range(warmup, meta['ticks']):
                recorded = inputs[f]
                status, packet = agent.tick(world.state)
                if status == 1: ended = True; break
                compared += 1
                rec = (int(recorded['angle'][agent_index]), int(recorded['buttons'][agent_index]))
                if rec == packet: matched += 1
                else:
                    if first_mismatch is None: first_mismatch = {'frame': f, 'episode_tick': f - warmup, 'recorded': rec, 'shim': packet}
                    if len(mismatches) < 10: mismatches.append((f, rec, packet))
                actions = [(int(recorded['angle'][p]), int(recorded['buttons'][p])) for p in range(len(players))]
                actions[agent_index] = packet if args.play_shim else rec
                world.step(actions)
                if world.ended:
                    status, _ = agent.tick(world.state); ended = status == 1; break
        finally:
            world.close()
    report = {'check': 'replay', 'replay': str(rep), 'replay_sha256': meta['sha256'], 'agent': spec, 'agent_id': agent_id,
              'seed': args.seed, 'agent_index': agent_index, 'level': meta['level'], 'frames': meta['ticks'], 'warmup': warmup,
              'compared': compared, 'matched': matched, 'first_mismatch': first_mismatch, 'mismatches': [str(m) for m in mismatches],
              'ended': ended, 'ok': compared > 0 and compared == matched}
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sim = sub.add_parser('sim'); sim.add_argument('agent'); sim.add_argument('--seed', type=int, default=1001)
    sim.add_argument('--maps', default='0,1,3'); sim.add_argument('--game', type=Path)
    sim.set_defaults(run=check_sim)
    rp = sub.add_parser('replay'); rp.add_argument('replay', type=Path); rp.add_argument('agent')
    rp.add_argument('--seed', type=int, required=True, help='round seed from the result JSON line')
    rp.add_argument('--agent-index', type=int, required=True, help='0-based index of the agent in the replay roster')
    rp.add_argument('--game', type=Path, help='Unity player the replay was recorded with (default BOPL_RL_GAME)')
    rp.add_argument('--play-shim', action='store_true', help='drive the world with the shim output instead of the recorded agent input')
    rp.set_defaults(run=check_replay)
    args = parser.parse_args()
    return args.run(args)

if __name__ == '__main__':
    sys.exit(main())
