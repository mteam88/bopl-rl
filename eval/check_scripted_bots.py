"""Scripted-bot sanity check: every kind must beat an idle opponent on maps 0-2.

Runs the bots (bopl_bots.h) through the research binding against a do-nothing
opponent, both hard checks (wins, no timeouts) and telemetry (arrows fired
and hit). No GPU, no torch; ~10 s. Kinds that need a forkable game (the
search bot) are skipped, with a note, when BOPL_RL_GAME is not the Rust sim.
Optionally pits bots against a checkpoint:

    check_scripted_bots.py                 # idle sanity gate (asserts)
    check_scripted_bots.py <checkpoint>    # report win rates vs a policy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import Environment, configure, selected_game
from evaluation_protocol import SCRIPTED_KINDS
from telemetry import FIELDS
import layout

GAMES_PER_MAP = 2  # one seed pair, both sides
SEED_BASE = 40001


def play(env, policy, kind, map_id, seed, bot_side, generator=None):
    import torch
    env.reset(seed, map_id)
    env.bot(0, 0); env.bot(1, 0)
    env.bot(bot_side, kind, 2)
    state = policy.initial(2, 'cpu') if policy else None
    other = 1 - bot_side
    stats = np.zeros(len(FIELDS))
    while not env.ended:
        actions = np.zeros((2, layout.NUM_ATNS), dtype=np.float32)
        if policy:
            from native_policy import decide
            with torch.no_grad():
                _, _, sampled, state = decide(policy, torch.from_numpy(env.obs.copy()), env.mask.copy(), state, generator)
            actions[other] = sampled[other].numpy()
        env.step(actions)
        stats += env.telemetry[bot_side]
    final = env.state
    timeout = bool(final.header.flags & 2)
    # Both sides eliminated on the same tick carries the win flag for everyone
    # and is a win for nobody, as the reward scores it (bopl_undecided in
    # train/bopl_puffer.h).
    undecided = len({p.team for p in final.players[:2] if p.flags & 2}) > 1
    win = bool(final.players[bot_side].flags & 2) and not timeout and not undecided
    return win, timeout, stats


def main():
    policy = None
    if len(sys.argv) > 1:
        from native_policy import load
        policy = load(sys.argv[1])[0]
        import torch
        torch.set_num_threads(1)
    configure(Path('/tmp/check-scripted-bots-logs'))
    failures = []
    with Environment(seed=90001, limit=7200, delay=9) as env:
        forks = env.forks()
        for kind_name, kind in SCRIPTED_KINDS.items():
            if Environment.bot_forks(kind) and not forks:
                print(f'{kind_name:8s} (hard): skipped, the game backend {selected_game()} has no bopl_rl_fork '
                      f'(run with BOPL_RL_GAME=<libbopl_sim.so>)', flush=True)
                continue
            import torch
            generator = torch.Generator().manual_seed(SEED_BASE) if policy else None
            wins = timeouts = n = 0
            stats = np.zeros(len(FIELDS))
            for map_id in (0, 1, 2):
                for game in range(GAMES_PER_MAP):
                    w, t, s = play(env, policy, kind, map_id,
                                   SEED_BASE + 100000 * map_id + game // 2, game % 2, generator)
                    wins += w; timeouts += t; n += 1; stats += s
            tele = dict(zip(FIELDS, stats))
            print(f'{kind_name:8s} (hard): {wins}/{n} wins, {timeouts} timeouts, '
                  f'arrows {tele["arrows_fired"]:.0f} hits {tele["arrow_hits"]:.0f}', flush=True)
            if policy is None and (wins != n or timeouts):
                failures.append(kind_name)
    if failures:
        raise SystemExit(f'FAIL: {", ".join(failures)} did not sweep the idle opponent')
    print('OK' if policy is None else 'report only (policy opponent)')


if __name__ == '__main__':
    main()
