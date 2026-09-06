"""Short Bow trials in the existing game, with exact research observations.

Task rewards are separate from full-game rewards. An accepted arrow hit is the
only success. Idle opponents can move with platforms/gravity; they are not frozen.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import numpy as np
from environment import Environment


@dataclass(frozen=True)
class Task:
    map_id: int = 0
    side: int = 0
    limit: int = 240
    assistance: str = 'none'
    hold_ticks: int = 64
    release_ticks: int = 16
    start_move_ticks: int = 0
    opponent: str = 'idle'

    def __post_init__(self):
        if self.side not in (0, 1) or self.limit < 4 or self.limit % 4:
            raise ValueError('Invalid side or four-tick trial limit')
        if self.assistance not in ('none', 'timing'):
            raise ValueError('Unknown assistance')
        if min(self.hold_ticks, self.release_ticks) < 4 or self.hold_ticks % 4 or self.release_ticks % 4:
            raise ValueError('Script intervals must be positive multiples of four ticks')
        if self.start_move_ticks % 4:
            raise ValueError('Starting movement must use whole four-tick decisions')
        if self.opponent not in ('idle', 'approach'):
            raise ValueError('Unknown opponent control')


class ChargeScript:
    """Press when ready, hold for a fixed interval, then release. DAP slot order."""
    def __init__(self, task):
        self.task = task
        self.counter = 0

    def buttons(self, observation):
        if self.counter == 0 and observation[10 + 16 + 44 + 42] > .5:
            self.counter = self.task.hold_ticks // 4
        if self.counter > 0:
            self.counter -= 1
            if self.counter == 0:
                self.counter = -self.task.release_ticks // 4
            return 4
        if self.counter < 0:
            self.counter += 1
        return 0


def point_at_opponent(observation):
    dx, dy = observation[10 + 148 + 9:10 + 148 + 11]
    return round(math.atan2(float(dy), float(dx)) % (2 * math.pi) * 255 / (2 * math.pi)) % 255 + 1


class Trial:
    def __init__(self, task):
        self.task = task
        self.env = Environment(seed=73, map_id=task.map_id, limit=task.limit + abs(task.start_move_ticks))
        self.done = True

    def reset(self, seed):
        self.env.reset(seed, self.task.map_id)
        # A reproducible physical setup variation, before the learner starts.
        # Positive ticks approach the opponent; negative ticks move away.
        for _ in range(abs(self.task.start_move_ticks) // 4):
            angle = point_at_opponent(self.observation())
            if self.task.start_move_ticks < 0:
                angle = (angle - 1 + 128) % 255 + 1
            actions = np.zeros((2, 2), np.int64)
            actions[self.task.side, 0] = angle
            self.env.step(actions)
            if any(p.deaths for p in self.env.state.players[:2]) or self.env.ended:
                raise RuntimeError('Starting movement ended the trial before learner control')
        self.seed = seed
        self.done = False
        self.script = ChargeScript(self.task)
        self.hold_ticks = 0
        self.completed_holds = []
        self.initial = self.env.frame()
        self.start_tick = self.env.state.header.episode_ticks
        return self.observation()

    def observation(self):
        return self.env.obs[self.task.side].copy()

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset the completed trial before stepping')
        chosen = np.asarray(action, dtype=np.int64).copy()
        if chosen.shape != (2,) or not 0 <= chosen[0] <= 255 or chosen[1] not in (0, 4):
            raise ValueError('Bow trials accept a stick code and Bow released/held only')
        if self.task.assistance == 'timing':
            chosen[1] = self.script.buttons(self.observation())
        if chosen[1] == 4:
            self.hold_ticks += 4
        elif self.hold_ticks:
            self.completed_holds.append(self.hold_ticks)
            self.hold_ticks = 0
        actions = np.zeros((2, 2), np.int64)
        actions[self.task.side] = chosen
        if self.task.opponent == 'approach':
            actions[1 - self.task.side, 0] = point_at_opponent(self.env.obs[1 - self.task.side])
        self.env.step(actions)
        state = self.env.state
        learner = state.players[self.task.side]
        opponent = state.players[1 - self.task.side]
        events = state.telemetry[self.task.side]
        success = events.arrow_hits > 0
        elapsed = state.header.episode_ticks - self.start_tick
        self.done = bool(success or learner.deaths or opponent.deaths or self.env.ended or elapsed >= self.task.limit)
        reward = 1.0 if success else -.25 if learner.deaths else 0.0
        result = None
        if self.done:
            result = dict(seed=self.seed, task=asdict(self.task), success=bool(success),
                          ticks=elapsed, game_ticks=state.header.episode_ticks, shots=events.arrows_fired,
                          hits=events.arrow_hits, deaths=learner.deaths,
                          target_deaths=opponent.deaths, reward=reward,
                          completed_hold_ticks=self.completed_holds.copy(),
                          censored_hold_ticks=self.hold_ticks,
                          timeout=bool(state.header.flags & 2),
                          initial_players=self.initial['players'],
                          final_players=self.env.frame()['players'])
        return self.observation(), reward, self.done, result, chosen

    def replay(self, path):
        self.env.replay(Path(path))

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
