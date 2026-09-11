"""Play Bopl in the browser against a checkpoint or a scripted bot, on the Rust sim.

Human calibration games: one real-time match at a time through the research
binding (environment.Environment), the opponent wired exactly as evaluate.py
wires it (4-tick decision cadence, 9-tick observation delay, latch and mask
rules, masked multinomial sampling), the human driven through the same
(aim byte, button bits, hold index) action interface from a keyboard/mouse
state the browser streams over a WebSocket. Every finished game appends a line
to <out>/games.jsonl and writes the BOPLREP2 replay (the writer evaluate.py
uses) plus the dashboard's physics-view JSON next to it. See PLAY.md.

Standard library only (no websockets package in the venv): a small HTTP/1.1
server with a hand-rolled RFC 6455 endpoint on one asyncio loop.
"""
import argparse
import asyncio
import base64
import datetime
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from environment import Environment, configure
from evaluate import masked, sample
from evaluation_protocol import SCRIPTED_NAMES, checkpoint_identity, scripted_opponent
import layout
from native_policy import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WEB = HERE / 'web'
RUST_SIM = ROOT / 'native/target/release/libbopl_sim.so'
# One decision every four ticks at Unity's fixed step (ProjectSettings/TimeManager.asset:
# Fixed Timestep 0.01666667 = 1/60 s; the env repeats each decision for 4 ticks, eval_binding.c research_open).
TICK_RATE = 60
DECISION_TICKS = 4
DECISION_PERIOD = DECISION_TICKS / TICK_RATE
TICK_PERIOD = 1 / TICK_RATE
FRAME_MODES = ('tick', 'decision')  # a frame every game tick (60/s) or every decision (15/s, client extrapolates)
DISPLAY_DELAYS = (0, 1, 2)  # optional display delay in decisions (client side), for parity tests with the policy's 9-tick delay
STATIC = {'/': 'play.html', '/play.html': 'play.html', '/play.js': 'play.js', '/arena.js': 'arena.js',
          '/play.css': 'play.css'}
MIME = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
        '.json': 'application/json', '.rep': 'application/octet-stream', '.jsonl': 'application/x-ndjson'}
# Browser key-state bit layout (play.js sends it): the four policy buttons in
# bopl_layout.h order, then the Unity keyboard movement actions W/A/S/D
# (Assets/MonoBehaviour/SlimeControls.asset, KeyboardAndMouse group).
KEY_JUMP, KEY_DASH, KEY_ARROW, KEY_PLATFORM, KEY_W, KEY_A, KEY_S, KEY_D = (1 << i for i in range(8))
BUTTON_MASK = KEY_JUMP | KEY_DASH | KEY_ARROW | KEY_PLATFORM
ABILITY_MASK = KEY_DASH | KEY_ARROW | KEY_PLATFORM
BINDINGS = [
    # Unity default KeyboardAndMouse bindings (Assets/MonoBehaviour/SlimeControls.asset)
    # plus keyboard alternates for the mouse buttons so a trackpad works.
    dict(action='Move', keys='W A S D / arrows', unity='W/A/S/D + arrows'),
    dict(action='Jump', keys='Space', unity='Space'),
    dict(action='Dash (ability 0)', keys='Left mouse / J', unity='Left mouse'),
    dict(action='Bow (ability 1)', keys='Right mouse / K', unity='Right mouse'),
    dict(action='Platform (ability 2)', keys='Middle mouse / L', unity='Middle mouse'),
    dict(action='Aim', keys='Mouse (mouse mode) / movement keys (keys mode)', unity='Mouse position'),
    dict(action='Bow charge', keys='1-5 or wheel = hold 1/2/4/8/16 decisions', unity='hold and release'),
]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def angle_byte(x, y):
    """Player.InputProfile.JoyStickAngleByteFromVector (Assets/Scripts/Assembly-CSharp/Player.cs:141):
    0 inside the deadzone, else int(angle/2pi*255) % 255 + 1 with angle in [0, 2pi) from +x.
    Constants.DEADZONE = 0.4 (Constants.cs:17); the keyboard directions are unit or zero vectors."""
    if x * x + y * y < 0.4 * 0.4:
        return 0
    angle = math.atan2(y, x) % (2 * math.pi)
    return int(angle / (2 * math.pi) * 255) % 255 + 1


class HumanInput:
    """Key state the browser streams; folded into one policy action per decision."""

    def __init__(self, mode='mouse'):
        self.down = 0        # currently held keys (KEY_* bits)
        self.pressed = 0     # press edges since the last decision (a tap inside one decision still counts)
        self.arrow_presses = 0  # arrow press edges since the last decision (edge triggered: a hold is one charge)
        self.aim = None      # mouse aim byte relative to the own body, or None
        self.charge = 4      # hold index for the next arrow press (layout.HOLD_LENGTHS)
        self.mode = mode     # 'mouse' or 'keys'
        self.seq = 0         # last input sequence number applied (latency measurement)
        self.latest_seq = 0

    def update(self, message):
        self.down = int(message.get('k', self.down)) & 0xFF
        pressed = int(message.get('p', 0)) & 0xFF
        self.pressed |= pressed
        if pressed & KEY_ARROW:
            self.arrow_presses += 1
        aim = message.get('a')
        self.aim = None if aim is None else max(0, min(255, int(aim)))
        if 'c' in message:
            self.charge = max(0, min(layout.ACT_HOLD - 1, int(message['c'])))
        if 's' in message:
            self.latest_seq = int(message['s'])

    def movement_byte(self, keys):
        x = bool(keys & KEY_D) - bool(keys & KEY_A)
        y = bool(keys & KEY_W) - bool(keys & KEY_S)
        return angle_byte(x, y)

    def decide(self, mask, active):
        """One (aim, buttons, hold, aim_source) decision under the environment's action mask.

        mask: layout.LOGITS bytes (1 = allowed) for this player's next decision, as
        bopl_write_mask published it. While a latch is in force only the held button
        state and hold length 1 are allowed (buttons are pinned by bopl_latch; aim
        stays live). active: an ability is in use this tick (bow charge, platform
        morph, dash) so mouse mode keeps aiming with the mouse."""
        latched = not mask[layout.ACT_AIM + layout.ACT_BUTTONS + 1]  # hold head reduced to index 0 <=> latched
        keys = self.down | self.pressed
        buttons = keys & BUTTON_MASK
        if self.arrow_presses:
            buttons |= KEY_ARROW
        else:
            buttons &= ~KEY_ARROW  # holding the bow key does not re-press: an arrow press is one charge
        hold = 0
        if latched:
            # Only the held button state is allowed: read it off the mask.
            allowed = [b for b in range(layout.ACT_BUTTONS) if mask[layout.ACT_AIM + b]]
            if len(allowed) != 1:
                raise RuntimeError(f'Latched mask allows {len(allowed)} button states')
            buttons = allowed[0]
        else:
            if not mask[layout.ACT_AIM + buttons]:
                # Arrow-release rule (bopl_write_mask): the decision after an arrow latch lets go.
                buttons &= ~KEY_ARROW
            if buttons & KEY_ARROW:
                hold = self.charge
        aim = self.movement_byte(keys)
        source = 'keys'
        if self.mode == 'mouse' and self.aim is not None and (latched or active or (buttons & ABILITY_MASK)):
            aim = self.aim
            source = 'mouse'
        if not mask[layout.ACT_AIM + buttons] or not mask[layout.ACT_AIM + layout.ACT_BUTTONS + hold]:
            raise RuntimeError(f'Human action violates the mask: buttons {buttons} hold {hold}')
        self.pressed = 0
        self.arrow_presses = 0
        self.seq = self.latest_seq
        return aim, buttons, hold, source, latched


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.identity = checkpoint_identity(self.path)
        self.meta = json.loads(Path(str(self.path) + '.json').read_text())
        self.step = int(self.meta.get('step', 0))
        self.id = self.identity['sha256'][:12]
        self.policy = None

    def label(self):
        """Short display name: the path below an artifacts/rl directory, without 'checkpoints/'."""
        parts = list(self.path.with_suffix('').parts)
        if 'artifacts' in parts:
            parts = parts[parts.index('artifacts') + 2:]
        return '/'.join(p for p in parts if p != 'checkpoints') or self.path.stem

    def load(self):
        if self.policy is None:
            self.policy, _ = load(self.path)
        return self.policy

    def describe(self):
        config = self.meta.get('config', {})
        return dict(kind='checkpoint', name=self.id, label=self.label(), path=str(self.path), step=self.step,
                    learner_steps=self.meta.get('learner_steps'), **self.identity,
                    episode_ticks=config.get('episode_ticks', 7200),
                    observation_delay=config.get('observation_delay_ticks', 9))


def list_checkpoints(patterns, latest_per_dir, known=None):
    """Checkpoint objects for the globs; `known` (path -> Checkpoint) is reused so a rescan
    while a sweep is still writing checkpoints only hashes the new files."""
    found = {}
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            if path.suffix != '.bin' or not Path(str(path) + '.json').is_file():
                continue
            found.setdefault(str(path.resolve()), path)
    if latest_per_dir:
        # Collapse policy-<step>.bin series to their newest step per directory; named anchors stay.
        best, keep = {}, {}
        for path in found.values():
            m = re.fullmatch(r'policy-(\d+)\.bin', path.name)
            if not m:
                keep[str(path.resolve())] = path; continue
            key = str(path.parent.resolve())
            if key not in best or int(m.group(1)) > best[key][0]:
                best[key] = (int(m.group(1)), path)
        found = {**keep, **{str(p.resolve()): p for _, p in best.values()}}
    checkpoints = []
    for key, path in found.items():
        if known and key in known:
            checkpoints.append(known[key]); continue
        try:
            checkpoints.append(Checkpoint(path))
        except Exception as error:  # a half-written sweep checkpoint is skipped, not fatal
            print(f'skip {path}: {error}', file=sys.stderr)
    checkpoints.sort(key=lambda c: str(c.path))
    return checkpoints


def slime_rows(env):
    """Per-slime ability readout for the HUD: owner, ability ids, cooldown fraction and
    flags (bopl_layout.h: 1 present, 2 ready, 4 over cooldown, 8 active, 0x70 bow charge, 0x80 fired)."""
    s = env.state
    fix = lambda v: v / 4294967296.
    rows = []
    for x in s.slimes[:s.header.slime_count]:
        rows.append([x.owner, list(x.abilities),
                     [round(min(1.0, fix(x.timers[i]) / fix(x.cooldowns[i])) if x.cooldowns[i] > 0 else 1.0, 3) for i in range(3)],
                     list(x.ability_flags)])
    return rows


def ability_active(env, player_id):
    s = env.state
    for x in s.slimes[:s.header.slime_count]:
        if x.owner == player_id:
            return any(f & 8 for f in x.ability_flags)
    return False


def compact_frame(frame, slimes):
    """Short-key frame for the wire; play.js expands it back to arena.js's shape."""
    r = lambda v: round(v, 3)
    return dict(t=frame['t'], g=frame['flags'],
                p=[[p['id'], p['flags'], r(p['x']), r(p['y']), r(p['scale']), p['kills'], p['deaths']] for p in frame['players']],
                e=[[e['kind'], e['layer'], e['owner'], r(e['x']), r(e['y']), r(e['rx']), r(e['ry']), r(e['ux']), r(e['uy']),
                    r(e['radius']), r(e['vx']), r(e['vy'])] for e in frame['entities']],
                s=slimes)


class Server:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.games_path = self.out / 'games.jsonl'
        self.checkpoints = []
        self.by_id = {}
        self.refresh()
        self.game = Path(args.game).resolve() if args.game else RUST_SIM
        if not self.game.exists():
            raise SystemExit(f'{self.game} is missing; run cargo build --release in native/')
        os.environ.setdefault('BOPL_RL_INPROC_SPARE', '0')
        configure(self.out / 'workers', self.game)
        self.game_sha256 = sha256_file(self.game)
        self.env = None
        self.env_params = None
        self.env_seed = None
        self.match = None
        self.games_played = 0
        self.loop = None

    def refresh(self):
        """Rescan the checkpoint globs (a running sweep keeps adding files); ids already offered stay valid."""
        known = {str(c.path): c for c in self.checkpoints}
        self.checkpoints = list_checkpoints(self.args.checkpoints, self.args.latest_per_dir, known)
        for c in self.checkpoints:
            self.by_id.setdefault(c.id, c)

    # ---- environment -------------------------------------------------------
    def environment(self, limit, delay):
        if self.env is None or self.env_params != (limit, delay):
            if self.env is not None:
                self.env.close()
            self.env_seed = 90001 + self.games_played
            self.env = Environment(seed=self.env_seed, map_id=0, limit=limit, delay=delay)
            self.env_params = (limit, delay)
        return self.env

    # ---- records -----------------------------------------------------------
    def games(self):
        if not self.games_path.exists():
            return []
        rows = []
        for line in self.games_path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    def tally(self, games=None):
        """Human record per opponent name: wins, losses, draws (timeouts) in Bradley-Terry form."""
        result = {}
        for g in games if games is not None else self.games():
            row = result.setdefault(g['b'], dict(wins=0, losses=0, draws=0, games=0, label=g.get('opponent', {}).get('label', g['b'])))
            row['games'] += 1
            row['wins' if g['outcome'] == 'a' else 'losses' if g['outcome'] == 'b' else 'draws'] += 1
        return result

    def lobby(self):
        self.refresh()
        games = self.games()
        return dict(
            checkpoints=[dict(id=c.id, label=c.label(), path=str(c.path), step=c.step, sha256=c.identity['sha256'])
                         for c in self.checkpoints],
            scripted=list(SCRIPTED_NAMES), maps=list(range(5)), tally=self.tally(games),
            games=games[-50:], bindings=BINDINGS, next_side=self.games_played % 2,
            game=dict(path=str(self.game), sha256=self.game_sha256), busy=self.match is not None,
            tick_rate=TICK_RATE, decision_ticks=DECISION_TICKS, hold_lengths=layout.HOLD_LENGTHS)

    # ---- HTTP ----------------------------------------------------------------
    async def handle(self, reader, writer):
        try:
            head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 10)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError):
            writer.close(); return
        lines = head.decode('latin-1').split('\r\n')
        try:
            method, target, _ = lines[0].split(' ', 2)
        except ValueError:
            writer.close(); return
        headers = {}
        for line in lines[1:]:
            if ':' in line:
                k, v = line.split(':', 1)
                headers[k.strip().lower()] = v.strip()
        path = target.split('?', 1)[0]
        query = target.split('?', 1)[1] if '?' in target else ''
        try:
            if headers.get('upgrade', '').lower() == 'websocket' and path == '/ws':
                await self.websocket(reader, writer, headers)
                return
            if method != 'GET':
                self.respond(writer, 405, b'method not allowed')
            elif path in STATIC:
                file = WEB / STATIC[path]
                self.respond(writer, 200, file.read_bytes(), MIME[file.suffix])
            elif path == '/api/lobby':
                self.respond(writer, 200, json.dumps(self.lobby()).encode(), MIME['.json'])
            elif path == '/api/games':
                self.respond(writer, 200, json.dumps(self.games()).encode(), MIME['.json'])
            elif path.startswith('/human/'):
                name = path[len('/human/'):]
                file = self.out / name
                if re.fullmatch(r'[A-Za-z0-9_.-]+', name) and file.is_file():
                    self.respond(writer, 200, file.read_bytes(), MIME.get(file.suffix, 'application/octet-stream'))
                else:
                    self.respond(writer, 404, b'not found')
            else:
                self.respond(writer, 404, b'not found')
        except Exception:
            traceback.print_exc()
            try:
                self.respond(writer, 500, b'error')
            except Exception:
                pass
        try:
            await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        writer.close()

    def respond(self, writer, status, body, mime='text/plain; charset=utf-8'):
        reason = {200: 'OK', 404: 'Not Found', 405: 'Method Not Allowed', 500: 'Internal Server Error'}[status]
        writer.write(f'HTTP/1.1 {status} {reason}\r\nContent-Type: {mime}\r\nContent-Length: {len(body)}\r\n'
                     f'Cache-Control: no-store\r\nConnection: close\r\n\r\n'.encode() + body)

    # ---- WebSocket (RFC 6455, server side, text frames) ----------------------
    async def websocket(self, reader, writer, headers):
        key = headers.get('sec-websocket-key')
        if not key:
            self.respond(writer, 400, b'bad websocket request'); return
        accept = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        writer.write(('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n'
                      f'Sec-WebSocket-Accept: {accept}\r\n\r\n').encode())
        await writer.drain()
        client = Client(self, reader, writer)
        await client.run()

    # ---- match lifecycle -------------------------------------------------------
    def opponent(self, spec):
        name = str(spec.get('opponent', ''))
        if name in self.by_id:
            return self.by_id[name]
        if scripted_opponent(name) is not None:
            return name
        raise ValueError(f'unknown opponent {name!r}')


def ws_frame(payload, opcode=1):
    if isinstance(payload, str):
        payload = payload.encode()
    n = len(payload)
    if n < 126:
        head = bytes([0x80 | opcode, n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 126]) + n.to_bytes(2, 'big')
    else:
        head = bytes([0x80 | opcode, 127]) + n.to_bytes(8, 'big')
    return head + payload


async def ws_read(reader):
    """One complete message: (opcode, payload). Fragmentation is not used by browsers for our small messages."""
    head = await reader.readexactly(2)
    opcode = head[0] & 0x0F
    masked_bit = head[1] & 0x80
    n = head[1] & 0x7F
    if n == 126:
        n = int.from_bytes(await reader.readexactly(2), 'big')
    elif n == 127:
        n = int.from_bytes(await reader.readexactly(8), 'big')
    key = await reader.readexactly(4) if masked_bit else b'\0\0\0\0'
    payload = await reader.readexactly(n)
    if masked_bit and n:
        # Unmask with a full-width XOR instead of a per-byte loop.
        repeated = (key * (n // 4 + 1))[:n]
        payload = (int.from_bytes(payload, 'little') ^ int.from_bytes(repeated, 'little')).to_bytes(n, 'little')
    return opcode, payload


class Client:
    def __init__(self, server, reader, writer):
        self.server = server
        self.reader = reader
        self.writer = writer
        self.input = HumanInput()
        self.match_task = None
        self.aborted = False
        self.closed = False
        self.dropped = 0
        self.bytes_sent = 0
        self.frames_sent = 0
        self.display = dict(frames='tick', frame_ticks=1, display_delay_decisions=0)

    def send(self, message):
        if self.closed:
            return False
        transport = self.writer.transport
        if transport.is_closing():
            return False
        # Never let a slow client stall the simulation: drop frames past 256 KB of unsent data.
        if transport.get_write_buffer_size() > 256 * 1024 and message.get('T') == 'f':
            self.dropped += 1
            return False
        payload = json.dumps(message, separators=(',', ':'))
        if message.get('T') == 'f':
            self.bytes_sent += len(payload); self.frames_sent += 1
        self.writer.write(ws_frame(payload))
        return True

    async def run(self):
        try:
            while True:
                try:
                    opcode, payload = await ws_read(self.reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                if opcode == 8:
                    break
                if opcode == 9:
                    self.writer.write(ws_frame(payload, 10)); continue
                if opcode != 1:
                    continue
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                kind = message.get('T')
                if kind == 'i':
                    self.input.update(message)
                elif kind == 'ping':
                    self.send(dict(T='pong', c=message.get('c'), s=time.time()))
                elif kind == 'start':
                    self.start(message)
                elif kind == 'abort':
                    self.aborted = True
        finally:
            self.closed = True
            self.aborted = True
            if self.match_task is not None and not self.match_task.done():
                try:
                    await self.match_task
                except Exception:
                    traceback.print_exc()
            self.writer.close()

    def start(self, spec):
        server = self.server
        if server.match is not None:
            self.send(dict(T='error', message='A match is already running; one at a time.')); return
        try:
            opponent = server.opponent(spec)
        except ValueError as error:
            self.send(dict(T='error', message=str(error))); return
        map_id = int(spec.get('map', 0))
        if not 0 <= map_id <= 30:
            self.send(dict(T='error', message='bad map')); return
        side = spec.get('side')
        side = server.games_played % 2 if side in (None, 'alternate') else int(side) & 1
        seed = int(spec['seed']) if spec.get('seed') not in (None, '') else secrets.randbits(31)
        self.input = HumanInput(mode='keys' if spec.get('mode') == 'keys' else 'mouse')
        frames = spec.get('frames', 'tick')
        display_delay = int(spec.get('display_delay', 0) or 0)
        if frames not in FRAME_MODES or display_delay not in DISPLAY_DELAYS:
            self.send(dict(T='error', message='bad frames/display_delay setting')); return
        self.display = dict(frames=frames, frame_ticks=1 if frames == 'tick' else DECISION_TICKS, display_delay_decisions=display_delay)
        self.aborted = False
        self.match_task = asyncio.get_running_loop().create_task(self.play(opponent, map_id, side, seed))

    async def play(self, opponent, map_id, side, seed):
        server = self.server
        server.match = self
        try:
            await self._play(opponent, map_id, side, seed)
        except Exception as error:
            traceback.print_exc()
            self.send(dict(T='error', message=f'match failed: {error}'))
        finally:
            server.match = None

    async def _play(self, opponent, map_id, side, seed):
        server = self.server
        loop = asyncio.get_running_loop()
        other = 1 - side
        scripted = None
        policy = None
        if isinstance(opponent, Checkpoint):
            description = opponent.describe()
            policy = opponent.load()
            limit, delay = description['episode_ticks'], description['observation_delay']
        else:
            scripted = scripted_opponent(opponent)
            description = dict(kind='scripted', name=opponent)
            limit, delay = 7200, 9
        env = server.environment(limit, delay)
        env.reset(seed, map_id)
        # Scripted bots decide inside the C environment from the true game state (SCRIPTED_BOTS.md "Contract").
        env.bot(0, 0); env.bot(1, 0)
        if scripted is not None:
            env.bot(other, *scripted)
        generator = torch.Generator().manual_seed(seed * 7 + side)
        state = policy.initial(2, 'cpu') if policy is not None else None
        started = time.time()
        frames = [env.frame()]
        initial = frames[0]
        human_id = initial['players'][side]['id']
        self.bytes_sent = 0; self.frames_sent = 0
        self.send(dict(T='start', side=side, map=map_id, seed=seed, opponent=description, mode=self.input.mode, display=self.display,
                       limit=limit, frame=compact_frame(initial, slime_rows(env)), hold_lengths=layout.HOLD_LENGTHS))
        per_tick = self.display['frames'] == 'tick'
        period = TICK_PERIOD if per_tick else DECISION_PERIOD
        latch_left = 0
        held = 0
        decisions = 0
        human_buttons = np.zeros(4, dtype=np.int64)
        hold_counts = np.zeros(layout.ACT_HOLD, dtype=np.int64)
        aim_sources = dict(mouse=0, keys=0)
        late = 0
        step_ms = []  # server work per decision (policy, input, sim ticks, frame encoding), sleeps excluded
        next_time = loop.time()

        async def pace():
            # Real time at Unity's fixed step (TimeManager.asset 1/60 s): one tick or one decision per period.
            nonlocal next_time, late
            next_time += period
            wait = next_time - loop.time()
            if wait < -2 * DECISION_PERIOD:
                # Fell behind by more than two decisions (scheduling hiccup): re-anchor instead of a speed burst.
                late += 1
                next_time = loop.time()
            elif wait > 0:
                await asyncio.sleep(wait)
            else:
                await asyncio.sleep(0)

        while not env.ended and not self.aborted:
            t0 = time.perf_counter()
            observation = torch.from_numpy(env.obs.copy())
            mask = env.mask.copy()
            actions = torch.zeros(2, layout.NUM_ATNS, dtype=torch.int64)
            if policy is not None:
                # Exactly evaluate.py: mask the logits as the trainer samples, multinomial per head.
                with torch.no_grad():
                    logits, _, state = policy.step(observation, state)
                    actions = sample(masked(logits, mask), generator)
            aim, buttons, hold, source, latched = self.input.decide(mask[side], ability_active(env, human_id))
            if latched != (latch_left > 0):
                raise RuntimeError('Latch bookkeeping diverged from the environment mask')
            aim_sources[source] += 1
            actions[side] = torch.tensor([aim, buttons, hold])
            # Mirror bopl_latch for the HUD: a free decision latches its buttons for HOLD_LENGTHS[hold] decisions.
            if latched:
                latch_left -= 1
            else:
                held = buttons; latch_left = layout.HOLD_LENGTHS[hold] - 1
                hold_counts[hold] += 1
            for bit in range(4):
                human_buttons[bit] += bool(held & (1 << bit))
            frames[-1]['actions'] = actions.tolist()
            decisions += 1
            hud = dict(s=self.input.seq, l=latch_left, b=held, a=aim, c=self.input.charge, m=self.input.mode)
            work = time.perf_counter() - t0
            if per_tick:
                # The decision's ticks one at a time (Environment.step_ticks: the same game as env.step, see
                # check_tick_step.py), each shown as soon as it is computed; the human's keys are still read once
                # per decision above and the policy keeps its 4-tick cadence and 9-tick delay.
                k = 0
                for left in env.step_ticks(actions.numpy()):
                    t1 = time.perf_counter()
                    k += 1
                    frame = env.frame()
                    self.send(dict(T='f', d=decisions, k=k, **compact_frame(frame, slime_rows(env)), h=hud))
                    work += time.perf_counter() - t1
                    if left:
                        await pace()
            else:
                env.step(actions.numpy())
                frame = env.frame()
                self.send(dict(T='f', d=decisions, k=DECISION_TICKS, **compact_frame(frame, slime_rows(env)), h=hud))
                work = time.perf_counter() - t0
            frames.append(frame)
            step_ms.append(work * 1e3)
            await pace()
        elapsed = time.time() - started
        if self.aborted or not env.ended:
            print(json.dumps(dict(aborted=True, opponent=description['name'], map=map_id, seed=seed, side=side,
                                  ticks=env.state.header.episode_ticks)), flush=True)
            self.send(dict(T='aborted'))
            return
        final = frames[-1]
        me, foe = final['players'][side], final['players'][other]
        timeout = bool(final['flags'] & 2)
        win = bool(me['flags'] & 2) and not timeout
        loss = bool(foe['flags'] & 2) and not timeout
        telemetry = final['telemetry']
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        name = f"{stamp}-{description['name']}-map{map_id}-s{seed}-p{side}"
        replay = server.out / (name + '.rep')
        env.replay(replay)
        visual = server.out / (name + '.json')
        record = dict(
            schema=1, time=started, timestamp=datetime.datetime.fromtimestamp(started).isoformat(timespec='seconds'),
            # Bradley-Terry row, elo_ladder.py's game shape: a = human, b = opponent, outcome a/b/draw.
            a='human', b=description['name'], map=map_id, seed=seed, side=side,
            outcome='a' if win else 'b' if loss else 'draw',
            result='win' if win else 'loss' if loss else 'timeout' if timeout else 'draw',
            win=win, loss=loss, timeout=timeout, ticks=final['t'], seconds=final['t'] / TICK_RATE,
            human=dict(side=side, kills=me['kills'], deaths=me['deaths'], arrows_fired=telemetry[side]['arrows_fired'],
                       arrow_hits=telemetry[side]['arrow_hits'], activations={k: telemetry[side][k + '_activations'] for k in ('dash', 'bow', 'platform')},
                       input_mode=self.input.mode, decisions=decisions,
                       button_rates=dict(zip(['jump', 'dash', 'arrow', 'platform'], (human_buttons / max(1, decisions)).tolist())),
                       hold_choices=dict(zip([str(n) for n in layout.HOLD_LENGTHS], hold_counts.tolist())),
                       aim_source_decisions=aim_sources),
            opponent=dict(description, kills=foe['kills'], deaths=foe['deaths'], arrows_fired=telemetry[other]['arrows_fired'],
                          arrow_hits=telemetry[other]['arrow_hits'], activations={k: telemetry[other][k + '_activations'] for k in ('dash', 'bow', 'platform')}),
            final_players=final['players'], replay=replay.name, replay_sha256=sha256_file(replay), visual=visual.name,
            game=dict(path=str(server.game), sha256=server.game_sha256), env_seed=server.env_seed,
            display=self.display,
            timing=dict(wall_seconds=elapsed, realtime_ratio=(final['t'] / TICK_RATE) / max(1e-9, elapsed),
                        step_ms_mean=float(np.mean(step_ms)) if step_ms else None, step_ms_max=float(np.max(step_ms)) if step_ms else None,
                        work_fraction=float(np.sum(step_ms)) / 1e3 / max(1e-9, elapsed),
                        reanchored=late, dropped_frames=self.dropped, frames_sent=self.frames_sent, frame_bytes=self.bytes_sent,
                        frame_bytes_per_frame=self.bytes_sent / max(1, self.frames_sent), frame_bytes_per_s=self.bytes_sent / max(1e-9, elapsed)),
            tick_rate=TICK_RATE, decision_ticks=DECISION_TICKS, observation_delay=delay if policy is not None else 9)
        # The dashboard's physics-view shape (evaluate.py): frames carry the actions applied after them.
        visual.write_text(json.dumps(dict(schema=1, step=description.get('step'), task=dict(opponent=description['name'], map=map_id,
                                          seed=seed, side=side, name=name), result=record, frames=frames,
                                          rendering='physics state replay', tick_rate=TICK_RATE, decision_rate=TICK_RATE // DECISION_TICKS,
                                          actions_timing='observation-before-action')))
        with server.games_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        server.games_played += 1
        print(json.dumps({k: record[k] for k in ('timestamp', 'b', 'map', 'seed', 'side', 'result', 'ticks', 'replay')}), flush=True)
        self.send(dict(T='end', record=record, tally=server.tally(), next_side=server.games_played % 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8891)
    parser.add_argument('--checkpoints', action='append', default=[], metavar='GLOB',
                        help='Native checkpoint .bin files to offer (repeatable glob)')
    parser.add_argument('--latest-per-dir', action='store_true', help='Keep only the highest policy-*.bin step per directory')
    parser.add_argument('--out', type=Path, default=ROOT / 'native/artifacts/rl/human', help='games.jsonl and replays')
    parser.add_argument('--game', type=Path, default=None, help=f'Sim library (default {RUST_SIM})')
    args = parser.parse_args()
    torch.set_num_threads(1)
    server = Server(args)
    print(f'{len(server.checkpoints)} checkpoints, {len(SCRIPTED_NAMES)} scripted bots; sim {server.game} ({server.game_sha256[:12]})', flush=True)

    try:  # this host's Tailscale address, for the printed URL only
        tailnet = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, timeout=5).stdout.split()[0]
    except Exception:
        tailnet = None

    async def serve():
        service = await asyncio.start_server(server.handle, args.host, args.port, limit=1 << 20)
        for sock in service.sockets:
            host, port = sock.getsockname()[:2]
            print(f'Bopl play: http://{host}:{port}/' + (f'  (tailnet: http://{tailnet}:{port}/)' if tailnet else ''), flush=True)
        async with service:
            await service.serve_forever()
    asyncio.run(serve())


if __name__ == '__main__':
    main()
