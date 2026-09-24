#!/usr/bin/env python3
"""Archive one play-vs-agent session (Tools/play-vs-agent.sh): copy every
round's auto-saved replay into the human results directory and append a game
row per round to games.jsonl.

Input: the JSON lines RlAgent.cs wrote to BOPL_AGENT_RESULTS (one per round).
Output rows are a superset of research/elo_ladder.py's play_games records
(a, b, map, seed, side, ticks, outcome) with a="human", b=<agent id>, so
tally()/Bradley-Terry can rate them like any other pairing:

  a            "human"
  b            "policy:<weights sha256>" or "bot:<kind>-<level>" (the shim's id)
  map          Unity level index the round was played on (0-123, the game's rotation)
  rl_map       the level's RL map index in the sim catalogue (native/data/map-set.json),
               null when the level is not ported; map_role = training | held-out |
               unported under the more-maps split, so the ladder can separate a fair
               human test (training maps) from a generalisation test (held-out maps)
  seed         round seed handed to the shim (bopl_agent_reset); replays the agent
  side         index of the human in the roster (0)
  ticks        episode ticks (after the 240-tick warmup)
  outcome      "a" (human won), "b" (agent won), "draw" (nobody / both)
  agent        the BOPL_AGENT spec as given, agent_path, agent_sha256 (checkpoint)
  roster       when the spec was a .roster: the roster path; agent/agent_path/
               agent_sha256/b then describe the member the shim drew for the
               round (joined on the round seed from --roster-log, the shim's
               {"seed","member","spec","id"} lines)
  human_abilities, agent_abilities   ability catalog indices
  replay       archived .rep path, replay_sha256, replay_source (game's copy)
  game_sha256, library_sha256, source_hash, catalog_hash, time, session
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
import map_set


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, help='BOPL_AGENT_RESULTS file of the session')
    parser.add_argument('--agent', required=True)
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--player-log', type=Path)
    parser.add_argument('--roster-log', type=Path, help="the shim's BOPL_AGENT_ROSTER_LOG (roster specs)")
    args = parser.parse_args()
    if not args.results.exists():
        print('no rounds were played (no results file)', file=sys.stderr)
        return 0
    rounds = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    catalogue = map_set.load()
    rl_of = map_set.rl_indices(catalogue)
    if not rounds:
        print('no rounds were played', file=sys.stderr)
        args.results.unlink()
        return 0
    session = args.results.stem.replace('.session-', '')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    roster_path = args.agent if args.agent.endswith('.roster') else None
    draws = {}
    if roster_path:
        log = args.roster_log or Path(roster_path + '.log')
        if not log.exists():
            print(f'roster session without its draw log {log}; rounds cannot be attributed', file=sys.stderr)
            return 1
        for line in log.read_text().splitlines():
            if line.strip():
                draw = json.loads(line)
                draws[int(draw['seed'])] = draw
    shas = {}

    def member(record):
        """(spec, id) deciding this round: the roster's draw, else the agent itself."""
        if not roster_path:
            return args.agent, record.get('agent_id')
        draw = draws.get(int(record.get('seed', -1)))
        return (draw['spec'], draw['id']) if draw else (None, None)

    def checkpoint(spec):
        path = Path(spec) if spec and spec.endswith('.bin') else None
        if path and path not in shas:
            shas[path] = sha256(path) if path.exists() else None
        return path, shas.get(path)

    game_sha = sha256(args.game) if args.game.exists() else None
    assembly = args.game.parent / 'BoplBattle_Data/Managed/Assembly-CSharp.dll'
    assembly_sha = sha256(assembly) if assembly.exists() else None
    library_sha = sha256(args.library) if args.library.exists() else None
    written = 0
    with open(out / 'games.jsonl', 'a') as games:
        for record in rounds:
            roster = record.get('players', [])
            humans = [p for p in roster if not p.get('agent')]
            agents = [p for p in roster if p.get('agent')]
            if not agents:
                continue
            agent = agents[0]
            human_index = next((i for i, p in enumerate(roster) if not p.get('agent')), 0)
            replay_source = record.get('replay')
            replay_copy = replay_sha = None
            if replay_source and Path(replay_source).exists():
                replay_copy = out / f"{session}-r{record.get('round', 0)}-{Path(replay_source).stem}.rep"
                shutil.copy2(replay_source, replay_copy)
                replay_sha = sha256(replay_copy)
            winner = record.get('winner')
            if record.get('draw') or winner is None:
                outcome = 'draw'
            elif winner == agent.get('id'):
                outcome = 'b'
            else:
                outcome = 'a'
            spec, agent_id = member(record)
            if spec is None:
                print(f"round {record.get('round')}: seed {record.get('seed')} has no roster draw; skipped", file=sys.stderr)
                continue
            agent_path, agent_sha = checkpoint(spec)
            level = record.get('level')
            row = {
                'a': 'human', 'b': agent_id, 'map': level, 'seed': record.get('seed'),
                'rl_map': rl_of.get(level), 'map_role': map_set.role_of_level(level, catalogue) if level is not None else None,
                'side': human_index, 'ticks': record.get('episode_ticks'), 'outcome': outcome,
                'agent': spec, 'agent_path': str(agent_path) if agent_path else None, 'agent_sha256': agent_sha,
                'roster': roster_path,
                'agent_index': record.get('agent_index'), 'agent_player': record.get('agent_player'),
                'human_abilities': [p.get('abilities') for p in humans], 'agent_abilities': agent.get('abilities'),
                'human_keyboard': [p.get('keyboard') for p in humans], 'players': roster,
                'warmup_ticks': record.get('warmup_ticks'), 'level_ticks': record.get('level_ticks'),
                'decisions': record.get('decisions'), 'stocks': record.get('stocks'), 'game_mode': record.get('game_mode'),
                'replay': str(replay_copy) if replay_copy else None, 'replay_sha256': replay_sha, 'replay_source': replay_source,
                'game': str(args.game), 'game_sha256': game_sha, 'assembly_sha256': assembly_sha,
                'library': str(args.library), 'library_sha256': library_sha,
                'source_hash': record.get('source_hash'), 'catalog_hash': record.get('catalog_hash'),
                'failed': record.get('failed', False), 'round': record.get('round'), 'session': session,
                'time': record.get('time'),
            }
            games.write(json.dumps(row) + '\n')
            written += 1
            print(f"round {record.get('round')}: map {row['map']} seed {row['seed']} outcome {outcome} "
                  f"({row['ticks']} ticks) vs {Path(spec).name if spec.endswith('.bin') else spec} replay {replay_copy}")
    if args.player_log and args.player_log.exists():
        shutil.copy2(args.player_log, out / f'{session}-Player.log')
    if roster_path and args.roster_log and args.roster_log.exists():
        shutil.copy2(args.roster_log, out / f'{session}-roster.log')
        args.roster_log.unlink()
    args.results.unlink()
    print(f"{written} game row(s) appended to {out / 'games.jsonl'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
