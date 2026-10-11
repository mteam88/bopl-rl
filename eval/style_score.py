"""Score a compiled style over replay dumps (docs/STYLES.md part 3, measurement).

  python eval/style_score.py styles/drill_snipe.toml [more.toml | style.bin ...] [--dumps DIR] [--json out.json]

A numpy interpreter of the same bytecode train/bopl_style.h runs: it decodes
style.bin (not the compiler's internals) and evaluates every expression over
all of a player's decisions at once, in float32 with the C operation order, so
each decision's per-term payout matches the C VM bit for bit (eval/style_check.c
holds the two to that on the dumps). Decisions are the ticks where
episode_ticks % 4 == 0, plus the last tick; an `events` table in a dump (or
synthetic events passed in) is delivered to the first decision at or after its
tick.

The report gives per-player-game rates for each term: the fraction of
decisions it paid on, payouts per minute, and its fire count per minute, split
by human winner, human loser (human-vs-human games with one winning team), and
the bot and human sides of human-vs-agent games (index.jsonl `agent`).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import style_compile as sc  # noqa: E402  (opcode tables and the style.bin decoder)

F32 = np.float32
NEVER = F32(1e9)
DECISION_TICKS = 4
EVENT_DTYPE = np.dtype([('tick', '<u4'), ('kind', '<u4'), ('subject', '<u4'), ('actor', '<u4'), ('entity', '<u4'),
                        ('source_ability', '<u4'), ('cause', '<u4'), ('flags', '<u4'), ('related', '<u4'),
                        ('related_actor', '<u4'), ('related_tick', '<u4'), ('x', '<f4'), ('y', '<f4'), ('vx', '<f4'),
                        ('vy', '<f4'), ('ax', '<f4'), ('ay', '<f4'), ('avx', '<f4'), ('avy', '<f4')])


def hypot(a, b):
    return np.sqrt(a * a + b * b)


def truth(v):
    return (v != 0).astype(F32)


def u32(v):
    """bs_u32: NaN/negative -> 0, >= 2^32 -> 0xffffffff, else truncate."""
    out = np.zeros(v.shape, np.uint64)
    ok = v >= 0
    big = ok & (v >= F32(4294967296.0))
    small = ok & ~big
    out[small] = v[small].astype(np.uint64)
    out[big] = 0xffffffff
    return out


def pseudo_angle(dx, dy):
    ax, ay = np.abs(dx), np.abs(dy)
    s = ax + ay
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(dy >= 0, np.where(dx >= 0, dy / s, F32(1) + ax / s),
                     np.where(dx < 0, F32(2) + ay / s, F32(3) + ax / s)).astype(F32)
    return np.where(s == 0, F32(0), a).astype(F32)


# ---- bytecode -> expression trees -------------------------------------------
class Node:
    __slots__ = ('op', 'arg', 'kids', 'sub', 'extra')

    def __init__(self, op, arg=0, kids=(), sub=None, extra=0):
        self.op, self.arg, self.kids, self.sub, self.extra = op, arg, list(kids), sub, extra


class Decoded:
    """A style program as expression trees, one per chunk entry."""
    def __init__(self, data):
        self.p = sc.decode(data)
        self.code = self.p['code']
        self.consts = np.array(self.p['consts'], dtype=F32)
        self.cats = self.p['cats']
        self.cache = {}
        self.slots = [dict(s, tree=self.chunk(s['pc'])) for s in self.p['slots']]
        self.terms = [dict(t, when_tree=self.chunk(t['when']), value_tree=self.chunk(t['value'])) for t in self.p['terms']]
        for k, slot in enumerate(self.slots):
            for ref in self.hist_refs(slot['tree']):
                if ref >= k:
                    raise ValueError(f'slot {k} reads slot {ref}, which is not updated before it')

    def chunk(self, pc):
        if pc not in self.cache:
            stack, end = self.run(pc, None)
            if len(stack) != 1:
                raise ValueError(f'chunk at {pc} leaves {len(stack)} values')
            self.cache[pc] = stack[0]
        return self.cache[pc]

    def run(self, pc, stop):
        stack = []
        while pc != stop:
            word = self.code[pc]
            name, arg = sc.OPS[word & 0xff], word >> 8
            if name == 'END':
                if stop is not None:
                    raise ValueError(f'jump target missed at {pc}')
                return stack, pc
            if name in ('JZ', 'JNZ'):
                cond = stack.pop()
                inner, _ = self.run(pc + 1, arg)
                if len(inner) != 1:
                    raise ValueError(f'unstructured jump at {pc}')
                stack.append(Node('AND' if name == 'JZ' else 'OR', kids=(cond, inner[0])))
                pc = arg
                continue
            if name == 'AGG':
                sub, extra = self.code[pc + 1], self.code[pc + 2]
                kind = sc.AGG_KINDS[arg]
                pops = {'MINDIST_ENT': 1, 'MINDIST_SLIME': 1, 'COVER_ENT': 2}.get(kind, 0)
                kids = stack[len(stack) - pops:] if pops else []
                del stack[len(stack) - pops:]
                stack.append(Node('AGG', arg, kids, self.chunk(sub), extra))
                pc += 3
                continue
            if name == 'LOOP':
                sub, var = self.code[pc + 1], self.code[pc + 2]
                stack.append(Node('LOOP', arg, (), self.chunk(sub), var))
                pc += 3
                continue
            pops, pushes = sc.EFFECT[name]
            kids = stack[len(stack) - pops:] if pops else []
            del stack[len(stack) - pops:]
            stack.append(Node(name, arg, kids))
            pc += 1
        return stack, pc

    def hist_refs(self, node, seen=None):
        out = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.op == 'HIST':
                out.add(n.arg)
            stack.extend(n.kids)
            if n.sub is not None:
                stack.append(n.sub)
        return out


# ---- game state --------------------------------------------------------------
PLAYER_FIELDS = sc.PLAYER_FIELDS
TELEM_FIELDS = sc.TELEMETRY_FIELDS
ENTITY_FIELDS = sc.ENTITY_FIELDS
SLIME_FIELDS = sc.SLIME_FIELDS
EVENT_FIELDS = sc.EVENT_FIELDS


class Game:
    """One dump's decision ticks as dense arrays."""
    def __init__(self, path, events=None):
        d = np.load(path)
        header = d['header']
        players, telemetry = d['players'], d['telemetry']
        self.P = int(players['player'].max()) + 1
        ticks = header['tick'].astype(np.int64)
        ep = header['episode_ticks'].astype(np.int64)
        rows = np.flatnonzero(ep % DECISION_TICKS == 0)
        if len(header) and (not len(rows) or rows[-1] != len(header) - 1):
            rows = np.append(rows, len(header) - 1)
        self.rows = rows
        self.D = len(rows)
        self.ticks = ticks[rows]
        self.header = {name: header[name][rows] for name in header.dtype.names}
        self.header['player_count'] = np.full(self.D, self.P, np.uint32)
        # Table rows carry `tick` = the header row index (eval/replay_dump.py's
        # loop counter); header.tick itself starts at 1.
        index = players['tick'].astype(np.int64)   # row tick = header row index (replay_dump's loop counter)
        keep = np.isin(index, rows)
        dec_of_row = np.full(len(header), -1)
        dec_of_row[rows] = np.arange(self.D)
        self.player = {}
        pd, pp = dec_of_row[index[keep]], players['player'][keep].astype(np.int64)
        for name in PLAYER_FIELDS:
            a = np.zeros((self.D, self.P), F32)
            a[pd, pp] = players[name][keep].astype(F32)
            self.player[name] = a
        self.player_raw = {name: np.zeros((self.D, self.P), players.dtype[name]) for name in PLAYER_FIELDS}
        for name in PLAYER_FIELDS:
            self.player_raw[name][pd, pp] = players[name][keep]
        tindex = telemetry['tick'].astype(np.int64)
        tkeep = np.isin(tindex, rows)
        td, tp = dec_of_row[tindex[tkeep]], telemetry['player'][tkeep].astype(np.int64)
        self.telem_raw = np.zeros((self.D, self.P, len(TELEM_FIELDS)), np.uint32)
        for k, name in enumerate(TELEM_FIELDS):
            self.telem_raw[td, tp, k] = telemetry[name][tkeep]
        self.telem = self.telem_raw.astype(F32)
        self.ent, self.ent_start, self.ent_count = self._rows(d['entities'], ticks, dec_of_row)
        self.sl, self.sl_start, self.sl_count = self._rows(d['slimes'], ticks, dec_of_row)
        self.ent_f = {name: self.ent[name].astype(F32) for name in ENTITY_FIELDS if name != 'speed'}
        self.ent_f['speed'] = hypot(self.ent_f['vx'], self.ent_f['vy'])
        self.sl_f = {name: self.sl[name].astype(F32) for name in SLIME_FIELDS if name not in ('exists', 'speed')}
        self.sl_f['speed'] = hypot(self.sl_f['vx'], self.sl_f['vy'])
        self.sl_abilities = self.sl['abilities'].astype(np.int64)
        self.sl_aflags = self.sl['ability_flags'].astype(np.int64)
        self.sl_arrays = [self.sl['abilities'].astype(F32), self.sl['ability_flags'].astype(F32),
                          self.sl['cooldowns'].astype(F32), self.sl['timers'].astype(F32)]
        if events is None and 'events' in d.files:
            events = d['events']
        self.set_events(events)
        self.body = self._bodies()
        self.meta = str(d['meta']) if 'meta' in d.files else ''

    def _rows(self, table, ticks, dec_of_row):
        dec = dec_of_row[table['tick'].astype(np.int64)]
        keep = dec >= 0
        rows = table[keep]
        dec = dec[keep]
        count = np.bincount(dec, minlength=self.D)
        start = np.concatenate([[0], np.cumsum(count)[:-1]])
        return rows, start, count

    def set_events(self, events):
        if events is None or len(events) == 0:
            self.ev = np.zeros(0, EVENT_DTYPE)
            self.ev_dec = np.zeros(0, np.int64)
        else:
            events = np.asarray(events)
            dec = np.searchsorted(self.ticks, events['tick'].astype(np.int64), side='left')
            keep = dec < self.D
            self.ev = events[keep]
            self.ev_dec = dec[keep]
        self.ev_count = np.bincount(self.ev_dec, minlength=self.D)
        self.ev_start = np.concatenate([[0], np.cumsum(self.ev_count)[:-1]])
        self.ev_f = {}
        for name in EVENT_FIELDS:
            if name == 'speed':
                self.ev_f[name] = hypot(self.ev['vx'].astype(F32), self.ev['vy'].astype(F32))
            elif name == 'aspeed':
                self.ev_f[name] = hypot(self.ev['avx'].astype(F32), self.ev['avy'].astype(F32))
            else:
                self.ev_f[name] = self.ev[name].astype(F32)

    def _bodies(self):
        """bs_body: per decision and player, the controlling slime row (-1 none)."""
        body = np.full((self.D, self.P), -1, np.int64)
        if not len(self.sl):
            return body
        dec = np.repeat(np.arange(self.D), self.sl_count)
        owner = self.sl['owner'].astype(np.int64)
        ab, af = self.sl_abilities, self.sl_aflags
        inventory = ((af & 1) != 0) & (ab > 0) & (ab <= 38)
        ok = inventory.any(1) & (owner >= 1) & (owner <= self.P)
        pi = np.clip(owner - 1, 0, self.P - 1)
        px, py = self.player['x'][dec, pi], self.player['y'][dec, pi]
        dx, dy = self.sl_f['x'] - px, self.sl_f['y'] - py
        dist = dx * dx + dy * dy
        pref = (self.sl['flags'] & 8) == 0
        idx = np.arange(len(self.sl))[ok]
        order = np.lexsort((idx, dist[ok], ~pref[ok], owner[ok], dec[ok]))
        idx = idx[order]
        key = dec[idx] * 8 + owner[idx]
        first = np.concatenate([[True], key[1:] != key[:-1]])
        idx = idx[first]
        body[dec[idx], owner[idx] - 1] = idx
        return body


# ---- evaluation --------------------------------------------------------------
class Ctx:
    __slots__ = ('d', 'ev', 'ent', 'sl', 'vars', 'L')

    def __init__(self, d, ev=None, ent=None, sl=None, vars=None):
        self.d, self.ev, self.ent, self.sl, self.vars = d, ev, ent, sl, dict(vars or {})
        self.L = len(d)

    def take(self, index, **changes):
        c = Ctx(self.d[index], None if self.ev is None else self.ev[index], None if self.ent is None else self.ent[index],
                None if self.sl is None else self.sl[index], self.vars)
        for key, value in changes.items():
            setattr(c, key, value)
        return c


class Player:
    """Evaluation of one program for one player of one game."""
    def __init__(self, prog, game, player, support):
        self.prog, self.g, self.self_index, self.support = prog, game, player, F32(1 if support else 0)
        self.self_id = F32(player + 1)
        g = game
        self.now = g.header['episode_ticks'].astype(F32)
        self.dt = np.empty(g.D, F32)
        if g.D:
            self.dt[0] = self.now[0]
            self.dt[1:] = self.now[1:] - self.now[:-1]
        self.opp = self._opponents()
        self.hist = []

    def _opponents(self):
        g, s = self.g, self.self_index
        best = np.zeros(g.D, F32)
        best_alive = np.zeros(g.D, bool)
        best_d = np.zeros(g.D, F32)
        sx, sy = g.player['x'][:, s], g.player['y'][:, s]
        team = g.player_raw['team']
        for i in range(g.P):
            if i == s:
                continue
            other = team[:, i] != team[:, s]
            alive = (g.player_raw['flags'][:, i] & 1) != 0
            dx, dy = g.player['x'][:, i] - sx, g.player['y'][:, i] - sy
            dist = dx * dx + dy * dy
            take = other & ((best == 0) | (alive & ~best_alive) | ((alive == best_alive) & (dist < best_d)))
            best = np.where(take, F32(i + 1), best)
            best_alive = np.where(take, alive, best_alive)
            best_d = np.where(take, dist, best_d)
        return best

    def ids(self, v):
        valid = (v >= 1) & (v <= self.g.P)
        return valid, np.where(valid, v, 1).astype(np.int64) - 1

    def pfield(self, v, d, table):
        valid, idx = self.ids(v)
        return np.where(valid, table[d, idx], F32(0)).astype(F32)

    def body_rows(self, v, d):
        valid, idx = self.ids(v)
        rows = self.g.body[d, idx]
        return np.where(valid, rows, -1)

    def slime_field(self, rows, f):
        name = SLIME_FIELDS[f]
        ok = rows >= 0
        if name == 'exists':
            return ok.astype(F32)
        src = self.g.sl_f[name]
        return np.where(ok, src[np.where(ok, rows, 0)] if len(src) else F32(0), F32(0)).astype(F32)

    def slime_array(self, rows, f, index):
        ok = (rows >= 0) & (index >= 0) & (index < 3)
        k = np.where(ok, index, 0).astype(np.int64)
        src = self.g.sl_arrays[f]
        if not len(src):
            return np.zeros(len(rows), F32)
        return np.where(ok, src[np.where(ok, rows, 0), k], F32(0)).astype(F32)

    def in_mask(self, cat, ability):
        if cat == sc.CAT_ANY:
            return ability > 0
        mask = np.uint64(self.prog.cats[cat])
        ok = (ability >= 0) & (ability < 64)
        a = np.where(ok, ability, 0).astype(np.uint64)
        return ok & (((mask >> a) & np.uint64(1)) != 0)

    def eval(self, n, c):
        op, g = n.op, self.g
        L = c.L
        if op == 'CONST':
            return np.full(L, self.prog.consts[n.arg], F32)
        if op == 'SELF':
            return np.full(L, self.self_id, F32)
        if op == 'OPP':
            return self.opp[c.d]
        if op == 'VAR':
            return np.full(L, F32(c.vars.get(n.arg, 0)), F32)
        if op == 'HEADER':
            return g.header[sc.HEADER_FIELDS[n.arg]][c.d].astype(F32)
        if op == 'DT':
            return self.dt[c.d]
        if op == 'EVSUP':
            return np.full(L, self.support, F32)
        if op == 'HIST':
            return self.hist[n.arg][c.d]
        if op == 'EVFIELD':
            if c.ev is None:
                return np.zeros(L, F32)
            return g.ev_f[EVENT_FIELDS[n.arg]][c.ev]
        if op == 'EFIELD':
            if c.ent is None:
                return np.zeros(L, F32)
            return g.ent_f[ENTITY_FIELDS[n.arg]][c.ent]
        if op == 'SFIELD':
            if c.sl is None:
                return np.zeros(L, F32)
            return self.slime_field(c.sl, n.arg)
        if op in ('AND', 'OR'):
            a = self.eval(n.kids[0], c)
            b = truth(self.eval(n.kids[1], c))
            if op == 'AND':
                return np.where(a == 0, F32(0), b).astype(F32)
            return np.where(a != 0, F32(1), b).astype(F32)
        if op == 'AGG':
            return self.aggregate(n, c)
        if op == 'LOOP':
            return self.loop(n, c)
        k = [self.eval(kid, c) for kid in n.kids]
        with np.errstate(all='ignore'):
            if op == 'PFIELD':
                return self.pfield(k[0], c.d, g.player[PLAYER_FIELDS[n.arg]])
            if op == 'TELEM':
                valid, idx = self.ids(k[0])
                return np.where(valid, g.telem[c.d, idx, n.arg], F32(0)).astype(F32)
            if op == 'BFIELD':
                return self.slime_field(self.body_rows(k[0], c.d), n.arg)
            if op == 'BFIELDI':
                return self.slime_array(self.body_rows(k[0], c.d), n.arg, k[1])
            if op == 'SFIELDI':
                if c.sl is None:
                    return np.zeros(L, F32)
                return self.slime_array(c.sl, n.arg, k[0])
            a = k[0]
            if op == 'ADD': return (a + k[1]).astype(F32)
            if op == 'SUB': return (a - k[1]).astype(F32)
            if op == 'MUL': return (a * k[1]).astype(F32)
            if op == 'DIV': return (a / k[1]).astype(F32)
            if op == 'LT': return (a < k[1]).astype(F32)
            if op == 'LE': return (a <= k[1]).astype(F32)
            if op == 'GT': return (a > k[1]).astype(F32)
            if op == 'GE': return (a >= k[1]).astype(F32)
            if op == 'EQ': return (a == k[1]).astype(F32)
            if op == 'NE': return (a != k[1]).astype(F32)
            if op == 'BAND': return (u32(a) & u32(k[1])).astype(F32)
            if op == 'BOR': return (u32(a) | u32(k[1])).astype(F32)
            if op == 'NOT': return (a == 0).astype(F32)
            if op == 'NEG': return (-a).astype(F32)
            if op == 'ABS': return np.abs(a)
            if op == 'SQRT': return np.sqrt(a)
            if op == 'MIN': return np.where(a < k[1], a, k[1]).astype(F32)
            if op == 'MAX': return np.where(a > k[1], a, k[1]).astype(F32)
            if op == 'HYPOT': return hypot(a, k[1])
            if op == 'TRUTH': return truth(a)
            if op in ('READY', 'HASAB', 'INFORM', 'FORMAB'):
                rows = self.body_rows(a, c.d)
                ok = rows >= 0
                r = np.where(ok, rows, 0)
                if not len(g.sl):
                    return np.zeros(L, F32)
                ab, af = g.sl_abilities[r], g.sl_aflags[r]
                if op in ('READY', 'HASAB'):
                    hit = self.in_mask(n.arg, ab)
                    if op == 'READY':
                        hit &= (af & 2) != 0
                    return np.where(ok, hit.sum(1), 0).astype(F32)
                form = (af & 8) != 0
                anyform = form.any(1)
                if op == 'INFORM':
                    return (ok & anyform).astype(F32)
                first = np.argmax(form, axis=1)
                return np.where(ok & anyform, ab[np.arange(L), first], 0).astype(F32)
            if op == 'INCAT':
                ok = (a >= 1) & (a < 64)
                return (ok & self.in_mask(n.arg, np.where(ok, a, 0).astype(np.int64))).astype(F32)
            if op == 'PDIST':
                va, ia = self.ids(a)
                vb, ib = self.ids(k[1])
                ax = np.where(va, g.player['x'][c.d, ia], F32(0)); ay = np.where(va, g.player['y'][c.d, ia], F32(0))
                bx = np.where(vb, g.player['x'][c.d, ib], F32(0)); by = np.where(vb, g.player['y'][c.d, ib], F32(0))
                return hypot((ax - bx).astype(F32), (ay - by).astype(F32))
            if op in ('EDIST', 'SDIST'):
                va, ia = self.ids(a)
                px = np.where(va, g.player['x'][c.d, ia], F32(0)); py = np.where(va, g.player['y'][c.d, ia], F32(0))
                rows = c.ent if op == 'EDIST' else c.sl
                if rows is None:
                    x = y = np.zeros(L, F32)
                else:
                    src = g.ent_f if op == 'EDIST' else g.sl_f
                    x, y = src['x'][rows], src['y'][rows]
                return hypot((x - px).astype(F32), (y - py).astype(F32))
        raise ValueError(f'unhandled op {op}')

    def expand(self, c, start, count):
        """Pairs (context row, table row) over the rows of each context row's decision."""
        cnt = count[c.d]
        l_idx = np.repeat(np.arange(c.L), cnt)
        offsets = np.concatenate([[0], np.cumsum(cnt)[:-1]]) if c.L else np.zeros(0, np.int64)
        rows = start[c.d][l_idx] + (np.arange(len(l_idx)) - offsets[l_idx])
        return l_idx, rows.astype(np.int64)

    def aggregate(self, n, c):
        g, kind = self.g, sc.AGG_KINDS[n.arg]
        kids = [self.eval(kid, c) for kid in n.kids]
        L = c.L
        if kind in ('COUNT_SLIME', 'MINDIST_SLIME'):
            l_idx, rows = self.expand(c, g.sl_start, g.sl_count)
            sub = c.take(l_idx, sl=rows)
        elif kind == 'COUNT_EV':
            l_idx, rows = self.expand(c, g.ev_start, g.ev_count)
            if n.extra:
                keep = g.ev['kind'][rows] == n.extra
                l_idx, rows = l_idx[keep], rows[keep]
            sub = c.take(l_idx, ev=rows)
        else:
            l_idx, rows = self.expand(c, g.ent_start, g.ent_count)
            sub = c.take(l_idx, ent=rows)
        m = self.eval(n.sub, sub) != 0
        li, ri = l_idx[m], rows[m]
        if kind in ('COUNT_ENT', 'COUNT_SLIME', 'COUNT_EV'):
            return np.bincount(li, minlength=L).astype(F32)
        if kind in ('MINDIST_ENT', 'MINDIST_SLIME'):
            src = g.ent_f if kind == 'MINDIST_ENT' else g.sl_f
            va, ia = self.ids(kids[0])
            px = np.where(va, g.player['x'][c.d, ia], F32(0)); py = np.where(va, g.player['y'][c.d, ia], F32(0))
            dist = hypot((src['x'][ri] - px[li]).astype(F32), (src['y'][ri] - py[li]).astype(F32))
            acc = np.full(L, NEVER, F32)
            np.fmin.at(acc, li, dist)
            return acc
        if kind == 'SOURCES_ENT':
            source = g.ent['source_ability'][ri].astype(np.int64)
            ok = source < 64
            bits = np.zeros(L, np.uint64)
            np.bitwise_or.at(bits, li[ok], np.left_shift(np.uint64(1), source[ok].astype(np.uint64)))
            return np.bitwise_count(bits).astype(F32)
        # COVER_ENT: kids = (player, radius)
        sectors, first = n.extra & 0xff, n.extra >> 8
        va, ia = self.ids(kids[0])
        px = np.where(va, g.player['x'][c.d, ia], F32(0)); py = np.where(va, g.player['y'][c.d, ia], F32(0))
        dx = (g.ent_f['x'][ri] - px[li]).astype(F32)
        dy = (g.ent_f['y'][ri] - py[li]).astype(F32)
        inside = hypot(dx, dy) <= kids[1][li]
        angle = pseudo_angle(dx, dy)
        sector = np.zeros(len(li), np.int64)
        for j in range(1, sectors):
            sector += angle >= self.prog.consts[first + j - 1]
        bits = np.zeros(L, np.uint64)
        np.bitwise_or.at(bits, li[inside], np.left_shift(np.uint64(1), sector[inside].astype(np.uint64)))
        return (np.bitwise_count(bits).astype(F32) / F32(sectors)).astype(F32)

    def loop(self, n, c):
        g, mode = self.g, sc.LOOP_MODES[n.arg]
        acc = np.full(c.L, {'all': 1.0, 'max': -1e9, 'min': 1e9}.get(mode, 0.0), F32)
        team = g.player_raw['team']
        s = self.self_index
        for i in range(g.P):
            include = team[c.d, i] != team[c.d, s]
            sub = Ctx(c.d, c.ev, c.ent, c.sl, {**c.vars, n.extra: i + 1})
            v = self.eval(n.sub, sub)
            with np.errstate(all='ignore'):
                if mode == 'sum': new = acc + v
                elif mode == 'any': new = np.where(v != 0, F32(1), acc)
                elif mode == 'all': new = np.where(v == 0, F32(0), acc)
                elif mode == 'max': new = np.where(v > acc, v, acc)
                elif mode == 'min': new = np.where(v < acc, v, acc)
                else: new = np.where(v != 0, acc + F32(1), acc)
            acc = np.where(include, new, acc).astype(F32)
        return acc

    def update_slots(self):
        g = self.g
        D = g.D
        base = Ctx(np.arange(D))
        now = self.now
        for slot in self.prog.slots:
            with np.errstate(all='ignore'):
                v = self.eval(slot['tree'], base)
            kind = sc.SLOT_NAMES[slot['kind']]
            out = np.empty(D, F32)
            if kind == 'prev':
                if D:
                    out[0] = v[0]
                    out[1:] = v[:-1]
            elif kind == 'since':
                last = np.where(v != 0, np.arange(D), -1)
                last = np.maximum.accumulate(last) if D else last
                has = last >= 0
                out = np.where(has, (now - now[np.where(has, last, 0)]).astype(F32), NEVER).astype(F32)
            elif kind == 'ema':
                tau = F32(slot['param'])
                h = F32(0)
                for j in range(D):
                    if j == 0:
                        h = v[0]
                    else:
                        a = F32(self.dt[j] / F32(tau + self.dt[j]))
                        h = F32(h + F32(a * F32(v[j] - h)))
                    out[j] = h
            else:
                cap, param = slot['cap'], F32(slot['param'])
                vals = truth(v) if kind == 'count_over' else v
                acc = np.full(D, F32(0) if kind == 'count_over' else F32(-1e9), F32)
                for back in range(cap):
                    i = np.arange(D) - back
                    ok = i >= 0
                    ic = np.where(ok, i, 0)
                    ok &= (now - now[ic]).astype(F32) < param
                    if kind == 'count_over':
                        acc = np.where(ok, (acc + vals[ic]).astype(F32), acc)
                    else:
                        acc = np.where(ok & (vals[ic] > acc), vals[ic], acc)
                out = acc.astype(F32)
            self.hist.append(out)

    def run(self):
        """Per-decision payout of every term (D x n_terms float32) and the total (D)."""
        g = self.g
        self.update_slots()
        D = g.D
        base = Ctx(np.arange(D))
        pays = np.zeros((D, len(self.prog.terms)), F32)
        with np.errstate(all='ignore'):
            for t, term in enumerate(self.prog.terms):
                if term['kind'] == 0:
                    when = self.eval(term['when_tree'], base)
                    value = self.eval(term['value_tree'], base)
                    pays[:, t] = np.where(when != 0, value, F32(0))
                else:
                    sel = np.flatnonzero(g.ev['kind'] == term['kind']) if len(g.ev) else np.zeros(0, np.int64)
                    if not len(sel):
                        continue
                    c = Ctx(g.ev_dec[sel], ev=sel)
                    when = self.eval(term['when_tree'], c)
                    value = self.eval(term['value_tree'], c)
                    fire = when != 0
                    col = np.zeros(D, F32)
                    np.add.at(col, g.ev_dec[sel][fire], value[fire])
                    pays[:, t] = col
        total = np.zeros(D, F32)
        for t in range(len(self.prog.terms)):
            total = (total + pays[:, t]).astype(F32)
        return pays, total


def score_game(prog, game, support=False):
    """{player index: (pays D x T, total D)}."""
    return {p: Player(prog, game, p, support).run() for p in range(game.P)}


# ---- grouping and report -----------------------------------------------------
def winners(game):
    flags = game.player_raw['flags'][-1]
    team = game.player_raw['team'][-1]
    won = {int(team[p]) for p in range(game.P) if flags[p] & 2}
    return won.pop() if len(won) == 1 else None


def groups_of(row, game):
    """[(player index, group)] for one index.jsonl row."""
    out = []
    if row.get('kind') == 'vs_agent' and row.get('agent'):
        agent = row['agent']
        bot = int(agent['agent_player']) - 1
        for p in range(game.P):
            out.append((p, 'bot' if p == bot else 'vs_agent_human'))
        return out
    team = winners(game)
    if team is None:
        return [(p, 'human_undecided') for p in range(game.P)]
    for p in range(game.P):
        out.append((p, 'human_winner' if int(game.player_raw['team'][-1, p]) == team else 'human_loser'))
    return out


GROUPS = ('human_winner', 'human_loser', 'bot', 'vs_agent_human')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('styles', nargs='+', type=Path, help='style TOMLs (compiled together) or one style.bin')
    parser.add_argument('--dumps', type=Path, default=ROOT / 'native/artifacts/human-replays/dump')
    parser.add_argument('--index', type=Path, default=ROOT / 'native/artifacts/human-replays/index.jsonl')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()
    if len(args.styles) == 1 and args.styles[0].suffix == '.bin':
        data, names = args.styles[0].read_bytes(), None
    else:
        compiled = sc.compile_files(args.styles)
        data, names = compiled.data, compiled.term_names
    prog = Decoded(data)
    names = names or [f'term{i}' for i in range(len(prog.terms))]
    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    per = {g: [] for g in GROUPS}
    games = 0
    for row in rows:
        path = args.dumps / (Path(row['file']).stem + '.npz')
        if not path.exists():
            continue
        game = Game(path)
        if game.D == 0:
            continue
        games += 1
        minutes = float(game.header['episode_ticks'][-1]) / 3600.0
        results = score_game(prog, game)
        for p, group in groups_of(row, game):
            if group not in per:
                continue
            pays, total = results[p]
            per[group].append(dict(file=row['file'], player=p + 1, minutes=minutes, decisions=game.D,
                                   pay=pays.astype(np.float64).sum(0).tolist(), fires=(pays != 0).sum(0).tolist()))
    table = []
    for t, name in enumerate(names):
        for group in GROUPS:
            items = per[group]
            if not items:
                continue
            frac = np.mean([it['fires'][t] / it['decisions'] for it in items])
            fpm = np.mean([it['fires'][t] / it['minutes'] for it in items])
            ppm = np.mean([it['pay'][t] / it['minutes'] for it in items])
            any_fire = np.mean([it['fires'][t] > 0 for it in items])
            table.append(dict(term=name, group=group, n=len(items), fire_fraction=float(frac), fires_per_min=float(fpm),
                              pay_per_min=float(ppm), games_with_fire=float(any_fire)))
    if args.json:
        args.json.write_text(json.dumps(dict(styles=[str(s) for s in args.styles], sha256=sc.hashlib.sha256(data).hexdigest(),
                                             games=games, table=table, players=per), indent=1))
    print(f'{games} games, program sha256 {sc.hashlib.sha256(data).hexdigest()[:16]}')
    print(f'| term | group | n | fire frac | fires/min | pay/min | games with fire |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for r in table:
        print(f"| {r['term']} | {r['group']} | {r['n']} | {r['fire_fraction']:.4f} | {r['fires_per_min']:.3f} | "
              f"{r['pay_per_min']:.5f} | {r['games_with_fire']:.3f} |")


if __name__ == '__main__':
    main()
