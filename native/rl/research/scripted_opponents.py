"""Stateful DAP opponents. Inputs are exclusively the learner's delayed features.

Each instance owns one player for one match. No game state, telemetry, map ID,
opponent actions or learner RNG is available to this interface.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PLAYER = 148
SELF = 10
OTHER = SELF + PLAYER
SLIMES = 10 + 6 * PLAYER + 96 * 21


@dataclass(frozen=True)
class BotSpec:
    name: str
    style: str
    reaction: int = 2
    aim_error: float = .025
    distance: float = 15.
    hold: int = 16
    recovery: int = 4
    lead: float = .65
    aggression: float = .5

    def __post_init__(self):
        if self.style not in ('archer', 'pursuer', 'evasive', 'switcher'):
            raise ValueError('Unknown bot style')
        if min(self.reaction, self.hold, self.recovery) < 1 or self.aim_error < 0:
            raise ValueError('Invalid bot timing or error')
        if not 0 <= self.aggression <= 1 or self.distance <= 0:
            raise ValueError('Invalid bot aggression or distance')


def load_specs(path):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 1:
        raise ValueError('Unknown script manifest schema')
    specs = [BotSpec(**item) for item in data['bots']]
    if not specs or len({s.name for s in specs}) != len(specs):
        raise ValueError('Scripts need unique identities')
    return specs


def identity(spec):
    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()


def angle_code(x, y):
    if abs(x) + abs(y) < 1e-8:
        return 0
    return round(math.atan2(y, x) % (2 * math.pi) * 255 / (2 * math.pi)) % 255 + 1


class ScriptedOpponent:
    def __init__(self, spec, seed):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.t = self.charge = self.wait = 0
        self.deaths = None
        self.view = None
        self.error = 0.
        self.stats = dict(charge_decisions=0, approach_decisions=0, retreat_decisions=0,
                          threats_seen=0, dodge_decisions=0)

    def act(self, observation):
        o = np.asarray(observation)
        if o.shape != (3257,) or not np.isfinite(o).all():
            raise ValueError('Expected one finite delayed DAP observation')
        # Reset a charge across a stock loss; match resets create a new object.
        deaths = round(float(o[SELF + 8]) * 10)
        if self.deaths != deaths:
            self.charge = self.wait = 0
            self.view = None
            self.deaths = deaths
        if self.view is None or self.t % self.spec.reaction == 0:
            self.view = o.copy()
        v = self.view
        self.t += 1
        if o[SELF + 3] < .5 or v[OTHER + 3] < .5 or v[OTHER + 12] > .5:
            self.charge = 0
            return np.zeros(2, dtype=np.int64)
        dx, dy = v[OTHER + 9:OTHER + 11] * 64
        distance = math.hypot(dx, dy)
        style = self.spec.style
        # Reserved behavior variant: changes tactic when the opponent charges.
        entities = v[898:2914].reshape(96, 21)
        # Projectile/reflectable layers, approaching within the next .4 s.
        # Feature 43 is a cooldown comparison, NOT an active-ability flag.
        threat = False
        for e in entities:
            if e[0] < .5 or e[7] > .5 or round(float(e[20])*32) not in (9, 14):
                continue
            position, velocity = e[8:10]*64, e[10:12]*256
            speed2 = float(velocity @ velocity)
            closest = float(np.clip(-(position @ velocity)/max(speed2, 1e-6), 0, .4))
            if closest > 0 and np.linalg.norm(position+velocity*closest) < 4:
                threat = True
                break
        if style == 'switcher':
            style = 'evasive' if threat else 'pursuer'
        self.stats['threats_seen'] += int(threat)
        ready = o[SELF + 16 + 44 + 42] > .5
        in_range = (style != 'pursuer' or distance < self.spec.distance) and not (style == 'evasive' and threat)
        if self.wait:
            self.wait -= 1
        if self.charge == 0 and not self.wait and ready and in_range:
            self.charge = self.spec.hold
            self.error = float(self.rng.normal(0, self.spec.aim_error))
        if self.charge:
            self.stats['charge_decisions'] += 1
            self.charge -= 1
            # Arrow speed approaches 75 world units/s at full charge. This is
            # an approximate aim heuristic, not an exact game-state solver.
            slimes = v[SLIMES:SLIMES + 12 * 28].reshape(12, 28)
            target = next((s for s in slimes if s[0] > .5 and s[2] < .5), None)
            vx, vy = target[9:11] * 256 if target is not None else (0., 0.)
            flight = min(distance / 65., .8)
            aim = math.atan2(dy + self.spec.lead * vy * flight + 4.91 * flight * flight + distance*.06,
                             dx + self.spec.lead * vx * flight) + self.error
            action = np.array([angle_code(math.cos(aim), math.sin(aim)), 4], dtype=np.int64)
            if self.charge == 0:
                self.wait = self.spec.recovery
            return action
        # Bow released while repositioning. Evasion has priority over approach.
        direction = 1 if dx >= 0 else -1
        move = direction
        if style == 'archer':
            move = direction if distance > self.spec.distance * 1.2 else -direction if distance < self.spec.distance * .7 else 0
        elif style == 'evasive':
            move = -direction if threat or distance < self.spec.distance else direction
        self.stats['approach_decisions'] += int(move == direction)
        self.stats['retreat_decisions'] += int(move == -direction)
        self.stats['dodge_decisions'] += int(style == 'evasive' and threat)
        # Stay within the arena; jump/platform help recover from low positions.
        x, y = o[8:10] * 64
        if abs(x) > 22:
            move = -1 if x > 0 else 1
        buttons = 0
        if self.t % 18 == 1 or y < -2:
            buttons |= 1
        if (style == 'evasive' and threat or style == 'pursuer' and distance > 9) and self.t % 12 == 1:
            if o[SELF + 16 + 42] > .5 and self.rng.random() < self.spec.aggression:
                buttons |= 2
        if y < -4 and o[SELF + 16 + 88 + 42] > .5 and self.t % 4 == 1:
            buttons |= 8
        return np.array([angle_code(move, .6 if y < -2 else 0), buttons], dtype=np.int64)
