"""Compile a style TOML (docs/STYLES.md part 2) into the bytecode train/bopl_style.h runs.

  python train/style_compile.py styles/drill_snipe.toml [-o run/style.bin] [--dump]

The output is a flat program: a float32 constants table, ability-category bit
masks, history slots (ring buffers the compiler lays out per player), and one
bytecode chunk per expression for a small stack VM. eval/style_score.py
interprets the same bytes offline; train/bopl_style.h runs them in the env.
Several files compile into one program (terms concatenated, names prefixed),
which is how the cost bench runs three styles at once.

Language (one expression per `when` / `value`):
  numbers, true/false, "strings" (category names), ( ), + - * /, & |,
  < <= > >= == !=, and/or/not (short-circuit), names below.
  self, opp           player ids: this player; the nearest living opponent
                      (nearest of any when none lives; 0 when there is none).
  p.<field>           id team flags alive_bodies stocks kills deaths cause_of_death
                      x y scale invisible_time ability_time, the telemetry
                      counters ability_activations[k] arrows_fired arrow_hits
                      deaths_environment deaths_opponent deaths_self deaths_other,
                      and derived alive won opponent_kills (kills - deaths_self).
  body(p).<field>     p's controlling slime: exists id owner flags state x y vx vy
                      rotation scale speed, and [slot] arrays abilities
                      ability_flags cooldowns timers.
  header.<field>      tick episode_ticks flags player_count entity_count
                      slime_count sudden_death.
  ev.<field>          event terms only: kind tick subject actor entity
                      source_ability cause flags related related_actor
                      related_tick x y vx vy ax ay avx avy speed aspeed.
  dt, now             ticks since this player's previous decision; episode_ticks.
  event_support       1 when the backend emits events (bopl_rl_event_support).
  ability.Drill, ability.Blink_gun ... ids; entity.EXPLOSIVE (or bare
  EXPLOSIVE) ... flags; slime.IN_ABILITY; death.RING_OUT; death_flag.RELATED_BLINKED;
  event.DEATH.
  min max abs sqrt hypot(a,b) clamp(x,lo,hi) dist(p,q) player(id)
  ready(p[, "cat"]) ready_count(p[, "cat"]) has_ability(p, "cat" | ability.X)
  in_category(id, "cat") in_form(p) form_ability(p) alive(p) since_form_entry(p)
  count|any|sources(entities|slimes|events|deaths|blinks [where <filter>])
  min_dist(entities|slimes where <filter>, p)
  coverage(p, radius, sectors, entities where <filter>)
  sum|any|all|max|min|count(for o in opponents: <expr>)
  since(cond) count_over(N, cond) max_over(N, expr) ema(tau, expr) prev(expr)
Inside a filter the row's fields are bare names (owner, flags, x, speed...) and
dist(p) is the row's distance to player p. History windows count ticks and are
sampled once per decision; their argument is a decision-level expression.
"""
import argparse
import hashlib
import math
import re
import struct
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b'BSTY'
VERSION = 1

# ---- bytecode (mirrored in train/bopl_style.h) -------------------------------
OPS = ['END', 'CONST', 'SELF', 'OPP', 'VAR', 'HEADER', 'PFIELD', 'BFIELD', 'BFIELDI', 'EVFIELD',
       'EFIELD', 'SFIELD', 'SFIELDI', 'ADD', 'SUB', 'MUL', 'DIV', 'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',
       'BAND', 'BOR', 'NOT', 'NEG', 'ABS', 'SQRT', 'MIN', 'MAX', 'HYPOT', 'JZ', 'JNZ', 'TRUTH',
       'READY', 'HASAB', 'INCAT', 'INFORM', 'FORMAB', 'PDIST', 'EDIST', 'SDIST', 'HIST', 'DT',
       'EVSUP', 'AGG', 'LOOP', 'TELEM']
OP = {name: i for i, name in enumerate(OPS)}
# Stack effect (pops, pushes) and extra words after the opcode word.
EFFECT = dict(END=(1, 0), CONST=(0, 1), SELF=(0, 1), OPP=(0, 1), VAR=(0, 1), HEADER=(0, 1), PFIELD=(1, 1),
              BFIELD=(1, 1), BFIELDI=(2, 1), EVFIELD=(0, 1), EFIELD=(0, 1), SFIELD=(0, 1), SFIELDI=(1, 1),
              NOT=(1, 1), NEG=(1, 1), ABS=(1, 1), SQRT=(1, 1), TRUTH=(1, 1), READY=(1, 1), HASAB=(1, 1),
              INCAT=(1, 1), INFORM=(1, 1), FORMAB=(1, 1), PDIST=(2, 1), EDIST=(1, 1), SDIST=(1, 1),
              HIST=(0, 1), DT=(0, 1), EVSUP=(0, 1), TELEM=(1, 1))
for _name in ('ADD', 'SUB', 'MUL', 'DIV', 'LT', 'LE', 'GT', 'GE', 'EQ', 'NE', 'BAND', 'BOR', 'MIN', 'MAX', 'HYPOT'):
    EFFECT[_name] = (2, 1)
EXTRA = dict(AGG=2, LOOP=2)
CAT_ANY = 0xFFFFFF

HEADER_FIELDS = ['tick', 'episode_ticks', 'flags', 'player_count', 'entity_count', 'slime_count', 'sudden_death']
PLAYER_FIELDS = ['id', 'team', 'flags', 'alive_bodies', 'stocks', 'kills', 'deaths', 'cause_of_death',
                 'x', 'y', 'scale', 'invisible_time', 'ability_time']
TELEMETRY_FIELDS = ['ability_activations0', 'ability_activations1', 'ability_activations2', 'arrows_fired',
                    'arrow_hits', 'deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other']
SLIME_FIELDS = ['exists', 'id', 'owner', 'flags', 'state', 'x', 'y', 'vx', 'vy', 'rotation', 'scale', 'speed']
SLIME_ARRAYS = ['abilities', 'ability_flags', 'cooldowns', 'timers']
ENTITY_FIELDS = ['id', 'kind', 'layer', 'owner', 'flags', 'hierarchy', 'source_ability', 'timer', 'x', 'y',
                 'vx', 'vy', 'right_x', 'right_y', 'up_x', 'up_y', 'radius', 'rotation', 'angular_velocity',
                 'inverse_mass', 'clock_rate', 'speed']
EVENT_FIELDS = ['kind', 'tick', 'subject', 'actor', 'entity', 'source_ability', 'cause', 'flags', 'related',
                'related_actor', 'related_tick', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'avx', 'avy', 'speed', 'aspeed']
PLAYER_TYPED = {'subject', 'actor', 'related_actor', 'owner'}

AGG_KINDS = ['COUNT_ENT', 'COUNT_SLIME', 'COUNT_EV', 'MINDIST_ENT', 'SOURCES_ENT', 'COVER_ENT', 'MINDIST_SLIME']
AGG = {name: i for i, name in enumerate(AGG_KINDS)}
LOOP_MODES = ['sum', 'any', 'all', 'max', 'min', 'count']
SLOT_KINDS = {'prev': 1, 'since': 2, 'count_over': 3, 'max_over': 4, 'ema': 5}
SLOT_NAMES = {v: k for k, v in SLOT_KINDS.items()}
HIDDEN_FLOATS = 2   # per player: [has a previous decision, its tick]
MAX_STACK, MAX_VARS, MAX_DEPTH = 32, 4, 3
LIMITS = dict(terms=32, slots=64, consts=1024, cats=32, code=1 << 16, hist_floats=4096)

EVENT_KINDS = {'death': 1, 'blink': 2}
DEATH_CAUSES = dict(UNKNOWN=0, CONTACT=1, PROJECTILE=2, EXPLOSION=3, BEAM=4, SQUISHED=5, EATEN=6,
                    RING_OUT=7, SELF=8, OTHER=9)
DEATH_FLAGS = dict(ACTOR_IN_FORM=1, LAST_BODY=2, RELATED_BLINKED=4)
ENTITY_FLAGS = dict(ACTIVE=1, ENABLED=2, DESTROYED=4, TRIGGER=8, COMPOSITE=16, DETACHED=32, TIMER_KNOWN=64,
                    PICKUP=128, PROJECTILE=256, SPARK_NODE=512, FLAMMABLE_SMOKE=1024, EXPLOSIVE=2048,
                    HITBOX=4096, CONTACT_DAMAGE=8192, CONTACT_DAMAGE_KNOWN=16384)
SLIME_FLAGS = dict(ACTIVE=1, SPAWNED=2, IN_ABILITY=4, SUPPRESS_INPUT=8, SIMULATED=16, ABILITYLESS=32)
PLAYER_FLAGS = dict(ALIVE=1, WON=2, INVISIBLE=4, TIME_STOP_PROTECTED=8, SPARK_SHOT=16, PLATFORM=32, MACHO_THROW=64)
ABILITY_FLAGS = dict(EQUIPPED=1, READY=2, FORM=8)
NAMESPACES = dict(entity=ENTITY_FLAGS, slime=SLIME_FLAGS, death=DEATH_CAUSES, death_flag=DEATH_FLAGS,
                  event={k.upper(): v for k, v in EVENT_KINDS.items()}, player_flag=PLAYER_FLAGS,
                  ability_flag=ABILITY_FLAGS)


def ability_table():
    path = ROOT / 'native/data/ability-coverage.json'
    import json
    rows = json.loads(path.read_text())['abilities']
    return {row['id']: row['name'] for row in rows if row['id'] > 0}


def canon(name):
    return re.sub(r'[\s_\-]', '', name).lower()


class CompileError(Exception):
    pass


# ---- tokenizer / parser -------------------------------------------------------
TOKEN = re.compile(r'\s*(?:(?P<num>\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)'
                   r'|(?P<str>"[^"]*"|\'[^\']*\')|(?P<name>[A-Za-z_][A-Za-z_0-9]*)'
                   r'|(?P<op><=|>=|==|!=|[-+*/()<>&|.,\[\]:]))')
KEYWORDS = {'and', 'or', 'not', 'where', 'for', 'in', 'true', 'false'}


class Node:
    def __init__(self, kind, pos, *args):
        self.kind, self.pos, self.args = kind, pos, args

    def __repr__(self):
        return f'{self.kind}{self.args}'


class Parser:
    def __init__(self, text, where):
        self.text, self.where, self.tokens = text, where, []
        pos = 0
        while pos < len(text):
            if text[pos:].strip() == '':
                break
            m = TOKEN.match(text, pos)
            if not m or m.end() == pos:
                self.fail(pos, f'unexpected character {text[pos:].lstrip()[:1]!r}')
            kind = m.lastgroup
            value = m.group(kind)
            start = m.start(kind)
            if kind == 'name' and value in KEYWORDS:
                kind = 'kw'
            self.tokens.append((kind, value, start))
            pos = m.end()
        self.tokens.append(('eof', None, len(text)))
        self.i = 0

    def fail(self, pos, message):
        raise CompileError(f'{self.where}: {message}\n    {self.text}\n    {" " * pos}^')

    def peek(self, offset=0):
        return self.tokens[min(self.i + offset, len(self.tokens) - 1)]

    def take(self):
        token = self.tokens[self.i]
        self.i += 1
        return token

    def accept(self, value):
        if self.peek()[1] == value and self.peek()[0] in ('op', 'kw'):
            return self.take()
        return None

    def expect(self, value):
        token = self.accept(value)
        if not token:
            kind, got, pos = self.peek()
            self.fail(pos, f'expected {value!r}, got {got if got is not None else "end of expression"!r}')
        return token

    def parse(self):
        node = self.expr()
        if self.peek()[0] != 'eof':
            self.fail(self.peek()[2], f'unexpected {self.peek()[1]!r}')
        return node

    def expr(self):
        node = self.and_()
        while (t := self.accept('or')):
            node = Node('or', t[2], node, self.and_())
        return node

    def and_(self):
        node = self.not_()
        while (t := self.accept('and')):
            node = Node('and', t[2], node, self.not_())
        return node

    def not_(self):
        if (t := self.accept('not')):
            return Node('not', t[2], self.not_())
        return self.cmp()

    def cmp(self):
        node = self.bitor()
        kind, value, pos = self.peek()
        if kind == 'op' and value in ('<', '<=', '>', '>=', '==', '!='):
            self.take()
            node = Node('bin', pos, value, node, self.bitor())
            if self.peek()[0] == 'op' and self.peek()[1] in ('<', '<=', '>', '>=', '==', '!='):
                self.fail(self.peek()[2], 'chained comparisons are not supported; join them with `and`')
        return node

    def bitor(self):
        node = self.bitand()
        while (t := self.accept('|')):
            node = Node('bin', t[2], '|', node, self.bitand())
        return node

    def bitand(self):
        node = self.add()
        while (t := self.accept('&')):
            node = Node('bin', t[2], '&', node, self.add())
        return node

    def add(self):
        node = self.mul()
        while self.peek()[0] == 'op' and self.peek()[1] in '+-':
            t = self.take()
            node = Node('bin', t[2], t[1], node, self.mul())
        return node

    def mul(self):
        node = self.unary()
        while self.peek()[0] == 'op' and self.peek()[1] in '*/':
            t = self.take()
            node = Node('bin', t[2], t[1], node, self.unary())
        return node

    def unary(self):
        if (t := self.accept('-')):
            return Node('neg', t[2], self.unary())
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while True:
            if (t := self.accept('.')):
                kind, value, pos = self.take()
                if kind not in ('name', 'kw'):
                    self.fail(pos, 'expected a field name after `.`')
                node = Node('attr', pos, node, value)
            elif (t := self.accept('[')):
                index = self.expr()
                self.expect(']')
                node = Node('index', t[2], node, index)
            else:
                return node

    def primary(self):
        kind, value, pos = self.take()
        if kind == 'num':
            return Node('num', pos, float(value))
        if kind == 'str':
            return Node('str', pos, value[1:-1])
        if kind == 'kw' and value in ('true', 'false'):
            return Node('num', pos, 1.0 if value == 'true' else 0.0)
        if kind == 'op' and value == '(':
            node = self.expr()
            self.expect(')')
            return node
        if kind == 'name':
            if self.accept('('):
                return Node('call', pos, value, self.call_args())
            return Node('name', pos, value)
        self.fail(pos, f'unexpected {value if value is not None else "end of expression"!r}')

    def call_args(self):
        args = []
        if self.accept(')'):
            return args
        while True:
            kind, value, pos = self.peek()
            if kind == 'kw' and value == 'for':
                self.take()
                var = self.take()
                if var[0] != 'name':
                    self.fail(var[2], 'expected a loop variable name after `for`')
                self.expect('in')
                coll = self.take()
                if coll[0] != 'name':
                    self.fail(coll[2], 'expected a collection after `in`')
                self.expect(':')
                args.append(Node('loop', pos, var[1], coll[1], self.expr()))
            elif kind == 'name' and value in COLLECTIONS and self.peek(1)[1] in ('where', ',', ')'):
                self.take()
                flt = self.expr() if self.accept('where') else None
                args.append(Node('coll', pos, value, flt))
            else:
                args.append(self.expr())
            if self.accept(')'):
                return args
            self.expect(',')


COLLECTIONS = {'entities': 'ent', 'slimes': 'slime', 'events': 'ev', 'deaths': 'ev', 'blinks': 'ev'}
COLLECTION_KIND = {'events': 0, 'deaths': EVENT_KINDS['death'], 'blinks': EVENT_KINDS['blink']}


# ---- code generation ---------------------------------------------------------
class Chunk:
    def __init__(self, label):
        self.label, self.words, self.refs, self.jumps = label, [], [], []

    def op(self, name, arg=0):
        if not 0 <= arg < (1 << 24):
            raise CompileError(f'internal: argument {arg} out of range for {name}')
        self.words.append(OP[name] | (arg << 8))

    def word(self, value):
        self.words.append(value)

    def ref(self, chunk):
        self.refs.append((len(self.words), chunk))
        self.words.append(0)

    def label_here(self):
        return len(self.words)


class Scope:
    """What names mean where an expression is compiled."""
    def __init__(self, domain='dec', event=False, vars=None, depth=0, history=True):
        self.domain, self.event, self.vars, self.depth, self.history = domain, event, dict(vars or {}), depth, history

    def child(self, **changes):
        scope = Scope(self.domain, self.event, self.vars, self.depth, self.history)
        for key, value in changes.items():
            setattr(scope, key, value)
        return scope


NUM, BOOL, PLAYER, BODY, STR, NS, EVENT = 'number', 'boolean', 'player', 'body', 'string', 'namespace', 'event'


class Compiler:
    def __init__(self, categories, abilities):
        self.consts, self.const_index = [], {}
        self.cats, self.cat_index = [], {}
        self.chunks, self.slots, self.terms = [], [], []
        self.hist_floats = HIDDEN_FLOATS
        self.decision_ticks = 4
        self.categories, self.abilities = categories, abilities
        self.ability_ids = {canon(name): i for i, name in abilities.items()}
        self.lets, self.where, self.text, self.expanding = {}, '', '', []

    # -- helpers
    def fail(self, node, message):
        pos = node.pos if node is not None else 0
        raise CompileError(f'{self.where}: {message}\n    {self.text}\n    {" " * pos}^')

    def const(self, value):
        value = float(np.float32(value))
        key = struct.pack('<f', value)
        if key not in self.const_index:
            self.const_index[key] = len(self.consts)
            self.consts.append(value)
        return self.const_index[key]

    def cat(self, mask):
        if mask not in self.cat_index:
            self.cat_index[mask] = len(self.cats)
            self.cats.append(mask)
        return self.cat_index[mask]

    def category_mask(self, node):
        if node.kind == 'str':
            name = node.args[0]
            if name not in self.categories:
                self.fail(node, f'unknown category {name!r} (known: {", ".join(sorted(self.categories))})')
            return self.categories[name]
        value = self.fold(node)
        if value is None or value != int(value) or not 0 < value < 64:
            self.fail(node, 'expected a category name in quotes or a constant ability id (ability.X)')
        return 1 << int(value)

    def ability_id(self, node, name):
        key = canon(name)
        if key not in self.ability_ids:
            self.fail(node, f'unknown ability {name!r} (known: {", ".join(n.replace(" ", "_") for n in self.abilities.values())})')
        return self.ability_ids[key]

    def fold(self, node):
        """The value of a constant expression, or None."""
        if node.kind == 'num':
            return node.args[0]
        if node.kind == 'neg':
            v = self.fold(node.args[0])
            return None if v is None else -v
        if node.kind == 'bin' and node.args[0] in '+-*/':
            a, b = self.fold(node.args[1]), self.fold(node.args[2])
            if a is None or b is None:
                return None
            return {'+': a + b, '-': a - b, '*': a * b, '/': a / b if b else math.inf}[node.args[0]]
        if node.kind == 'attr' and node.args[0].kind == 'name':
            space, field = node.args[0].args[0], node.args[1]
            if space == 'ability':
                return float(self.ability_id(node, field))
            if space in NAMESPACES and field in NAMESPACES[space]:
                return float(NAMESPACES[space][field])
        if node.kind == 'name' and node.args[0] in ENTITY_FLAGS:
            return float(ENTITY_FLAGS[node.args[0]])
        if node.kind == 'name' and node.args[0] in self.lets:
            return self.fold(self.lets[node.args[0]])
        return None

    def expect_value(self, node, t, what):
        if t in (STR, NS, EVENT, BODY):
            self.fail(node, f'{what} needs a number, got a {t}')

    # -- expressions
    def gen(self, node, chunk, scope):
        k = node.kind
        if k == 'num':
            chunk.op('CONST', self.const(node.args[0]))
            return NUM
        if k == 'str':
            self.fail(node, 'a string is only valid as a category argument (ready, ready_count, has_ability, in_category)')
        if k == 'name':
            return self.gen_name(node, chunk, scope)
        if k == 'attr':
            return self.gen_attr(node, chunk, scope)
        if k == 'index':
            return self.gen_index(node, chunk, scope)
        if k == 'call':
            return self.gen_call(node, chunk, scope)
        if k in ('and', 'or'):
            t = self.gen(node.args[0], chunk, scope)
            self.expect_value(node.args[0], t, f'`{k}`')
            jump = len(chunk.words)
            chunk.op('JZ' if k == 'and' else 'JNZ')
            t = self.gen(node.args[1], chunk, scope)
            self.expect_value(node.args[1], t, f'`{k}`')
            chunk.op('TRUTH')
            chunk.jumps.append((jump, len(chunk.words)))
            return BOOL
        if k == 'not':
            t = self.gen(node.args[0], chunk, scope)
            self.expect_value(node.args[0], t, '`not`')
            chunk.op('NOT')
            return BOOL
        if k == 'neg':
            t = self.gen(node.args[0], chunk, scope)
            if t != NUM and t != BOOL:
                self.fail(node, f'cannot negate a {t}')
            chunk.op('NEG')
            return NUM
        if k == 'bin':
            op, a, b = node.args
            ta = self.gen(a, chunk, scope)
            tb = self.gen(b, chunk, scope)
            if op in ('==', '!='):
                for side, t in ((a, ta), (b, tb)):
                    self.expect_value(side, t, f'`{op}`')
            elif op in ('<', '<=', '>', '>='):
                for side, t in ((a, ta), (b, tb)):
                    if t not in (NUM, BOOL):
                        self.fail(side, f'`{op}` compares numbers, got a {t}')
            else:
                for side, t in ((a, ta), (b, tb)):
                    if t not in (NUM, BOOL):
                        hint = ' (use p.id for the number)' if t == PLAYER else ''
                        self.fail(side, f'`{op}` needs numbers, got a {t}{hint}')
            chunk.op({'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '<': 'LT', '<=': 'LE', '>': 'GT',
                      '>=': 'GE', '==': 'EQ', '!=': 'NE', '&': 'BAND', '|': 'BOR'}[op])
            return BOOL if op in ('<', '<=', '>', '>=', '==', '!=') else NUM
        if k in ('coll', 'loop'):
            self.fail(node, 'a collection or `for` loop is only valid as an aggregate argument')
        self.fail(node, f'internal: unknown node {k}')

    def gen_name(self, node, chunk, scope):
        name = node.args[0]
        if name in scope.vars:
            chunk.op('VAR', scope.vars[name])
            return PLAYER
        if name == 'self':
            chunk.op('SELF')
            return PLAYER
        if name == 'opp':
            chunk.op('OPP')
            return PLAYER
        if name == 'dt':
            chunk.op('DT')
            return NUM
        if name == 'now':
            chunk.op('HEADER', HEADER_FIELDS.index('episode_ticks'))
            return NUM
        if name == 'event_support':
            chunk.op('EVSUP')
            return NUM
        if name == 'ev':
            if not scope.event:
                self.fail(node, '`ev` exists only in an event term (on = "death") or an events filter')
            return EVENT
        if scope.domain == 'ent' and name in ENTITY_FIELDS:
            chunk.op('EFIELD', ENTITY_FIELDS.index(name))
            return PLAYER if name in PLAYER_TYPED else NUM
        if scope.domain == 'slime' and name in SLIME_FIELDS and name != 'exists':
            chunk.op('SFIELD', SLIME_FIELDS.index(name))
            return PLAYER if name in PLAYER_TYPED else NUM
        if name in ENTITY_FLAGS:
            chunk.op('CONST', self.const(ENTITY_FLAGS[name]))
            return NUM
        if name in self.lets:
            if name in self.expanding:
                self.fail(node, f'let {name!r} refers to itself')
            self.expanding.append(name)
            saved = self.where, self.text
            self.where, self.text = f'{saved[0]} (in let {name})', self.let_text[name]
            t = self.gen(self.lets[name], chunk, scope)
            self.where, self.text = saved
            self.expanding.pop()
            return t
        if name in ('header', 'ability') or name in NAMESPACES:
            return NS
        known = ['self', 'opp', 'dt', 'now', 'event_support', 'header', 'ability'] + sorted(NAMESPACES) + sorted(self.lets)
        if scope.event:
            known.append('ev')
        known += list(scope.vars)
        if scope.domain == 'ent':
            known += ENTITY_FIELDS
        if scope.domain == 'slime':
            known += SLIME_FIELDS[1:]
        if name in ENTITY_FIELDS or name in SLIME_FIELDS:
            self.fail(node, f'{name!r} is a row field: it exists only inside an entities/slimes filter')
        self.fail(node, f'unknown name {name!r} (known here: {", ".join(known)})')

    def gen_attr(self, node, chunk, scope):
        obj, field = node.args
        if obj.kind == 'name' and obj.args[0] == 'ability' and obj.args[0] not in scope.vars:
            chunk.op('CONST', self.const(self.ability_id(node, field)))
            return NUM
        if obj.kind == 'name' and obj.args[0] in NAMESPACES and obj.args[0] not in scope.vars:
            table = NAMESPACES[obj.args[0]]
            if field not in table:
                self.fail(node, f'{obj.args[0]} has no {field!r} (known: {", ".join(table)})')
            chunk.op('CONST', self.const(table[field]))
            return NUM
        if obj.kind == 'name' and obj.args[0] == 'header' and obj.args[0] not in scope.vars:
            if field not in HEADER_FIELDS:
                self.fail(node, f'header has no {field!r} (known: {", ".join(HEADER_FIELDS)})')
            chunk.op('HEADER', HEADER_FIELDS.index(field))
            return NUM
        t = self.gen(obj, chunk, scope)
        if t == EVENT:
            if field not in EVENT_FIELDS:
                self.fail(node, f'ev has no {field!r} (known: {", ".join(EVENT_FIELDS)})')
            chunk.op('EVFIELD', EVENT_FIELDS.index(field))
            return PLAYER if field in PLAYER_TYPED else NUM
        if t == PLAYER:
            if field in PLAYER_FIELDS:
                chunk.op('PFIELD', PLAYER_FIELDS.index(field))
                return NUM
            if field in TELEMETRY_FIELDS:
                chunk.op('TELEM', TELEMETRY_FIELDS.index(field))
                return NUM
            if field == 'alive':
                chunk.op('PFIELD', PLAYER_FIELDS.index('flags'))
                chunk.op('CONST', self.const(1))
                chunk.op('BAND')
                return BOOL
            if field == 'won':
                chunk.op('PFIELD', PLAYER_FIELDS.index('flags'))
                chunk.op('CONST', self.const(2))
                chunk.op('BAND')
                chunk.op('TRUTH')
                return BOOL
            if field == 'opponent_kills':
                # the player id is on the stack once; kills - deaths_self reads it twice
                chunk.op('PFIELD', PLAYER_FIELDS.index('kills'))
                self.gen(obj, chunk, scope)
                chunk.op('TELEM', TELEMETRY_FIELDS.index('deaths_self'))
                chunk.op('SUB')
                return NUM
            if field == 'ability_activations':
                self.fail(node, 'ability_activations needs a constant [slot]')
            known = PLAYER_FIELDS + [f for f in TELEMETRY_FIELDS if not f.startswith('ability_activations')] + \
                ['ability_activations[k]', 'alive', 'won', 'opponent_kills']
            self.fail(node, f'a player has no {field!r} (known: {", ".join(known)})')
        if t == BODY:
            if field in SLIME_ARRAYS:
                self.fail(node, f'body(p).{field} is per slot: write body(p).{field}[slot]')
            if field not in SLIME_FIELDS:
                self.fail(node, f'a body has no {field!r} (known: {", ".join(SLIME_FIELDS + [a + "[slot]" for a in SLIME_ARRAYS])})')
            chunk.op('BFIELD', SLIME_FIELDS.index(field))
            return PLAYER if field == 'owner' else NUM
        self.fail(node, f'a {t} has no fields')

    def gen_index(self, node, chunk, scope):
        obj, index = node.args
        if obj.kind == 'attr':
            base, field = obj.args
            if field == 'ability_activations':
                slot = self.fold(index)
                if slot not in (0, 1, 2):
                    self.fail(index, 'ability_activations[k] needs a constant k of 0, 1 or 2')
                t = self.gen(base, chunk, scope)
                if t != PLAYER:
                    self.fail(base, f'ability_activations belongs to a player, got a {t}')
                chunk.op('TELEM', int(slot))
                return NUM
            if field in SLIME_ARRAYS:
                t = self.gen(base, chunk, scope)
                if t != BODY:
                    self.fail(base, f'{field}[slot] belongs to body(p), got a {t}')
                ti = self.gen(index, chunk, scope)
                self.expect_value(index, ti, 'a slot index')
                chunk.op('BFIELDI', SLIME_ARRAYS.index(field))
                return NUM
        if obj.kind == 'name' and obj.args[0] in SLIME_ARRAYS and scope.domain == 'slime':
            ti = self.gen(index, chunk, scope)
            self.expect_value(index, ti, 'a slot index')
            chunk.op('SFIELDI', SLIME_ARRAYS.index(obj.args[0]))
            return NUM
        self.fail(node, 'only ability_activations[k], body(p).<array>[slot] and, in a slimes filter, <array>[slot] take an index')

    def player_arg(self, node, chunk, scope, what):
        t = self.gen(node, chunk, scope)
        if t != PLAYER:
            self.fail(node, f'{what} needs a player (self, opp, a loop variable, ev.actor, player(id)), got a {t}')

    def nargs(self, node, args, lo, hi=None):
        hi = lo if hi is None else hi
        if not lo <= len(args) <= hi:
            want = str(lo) if lo == hi else f'{lo} to {hi}'
            self.fail(node, f'{node.args[0]}() takes {want} argument(s), got {len(args)}')

    def gen_call(self, node, chunk, scope):
        name, args = node.args
        if name in LOOP_MODES and args and args[0].kind == 'loop':
            return self.gen_loop(node, name, args, chunk, scope)
        if name in ('min', 'max'):
            if len(args) < 2:
                self.fail(node, f'{name}() takes at least 2 arguments')
            for i, arg in enumerate(args):
                if arg.kind == 'coll':
                    self.fail(arg, f'{name}() over a collection: use min_dist, or {name}(for o in opponents: ...)')
                if arg.kind == 'loop':
                    if len(args) != 1:
                        self.fail(arg, f'{name}(for ...) takes only the loop')
                t = self.gen(arg, chunk, scope)
                self.expect_value(arg, t, f'{name}()')
                if i:
                    chunk.op(name.upper())
            return NUM
        if name in ('abs', 'sqrt'):
            self.nargs(node, args, 1)
            t = self.gen(args[0], chunk, scope)
            self.expect_value(args[0], t, f'{name}()')
            chunk.op(name.upper())
            return NUM
        if name == 'hypot':
            self.nargs(node, args, 2)
            for arg in args:
                self.expect_value(arg, self.gen(arg, chunk, scope), 'hypot()')
            chunk.op('HYPOT')
            return NUM
        if name == 'clamp':
            self.nargs(node, args, 3)
            for i, arg in enumerate(args):
                self.expect_value(arg, self.gen(arg, chunk, scope), 'clamp()')
                if i == 1:
                    chunk.op('MAX')
                if i == 2:
                    chunk.op('MIN')
            return NUM
        if name == 'player':
            self.nargs(node, args, 1)
            self.expect_value(args[0], self.gen(args[0], chunk, scope), 'player()')
            return PLAYER
        if name == 'body':
            self.nargs(node, args, 1)
            self.player_arg(args[0], chunk, scope, 'body()')
            return BODY
        if name == 'alive':
            self.nargs(node, args, 1)
            self.player_arg(args[0], chunk, scope, 'alive()')
            chunk.op('PFIELD', PLAYER_FIELDS.index('flags'))
            chunk.op('CONST', self.const(1))
            chunk.op('BAND')
            return BOOL
        if name == 'dist':
            if len(args) == 1:
                if scope.domain not in ('ent', 'slime'):
                    self.fail(node, 'dist(p) is the row\'s distance inside an entities/slimes filter; elsewhere write dist(a, b)')
                self.player_arg(args[0], chunk, scope, 'dist()')
                chunk.op('EDIST' if scope.domain == 'ent' else 'SDIST')
                return NUM
            self.nargs(node, args, 2)
            self.player_arg(args[0], chunk, scope, 'dist()')
            self.player_arg(args[1], chunk, scope, 'dist()')
            chunk.op('PDIST')
            return NUM
        if name in ('ready', 'ready_count', 'has_ability'):
            self.nargs(node, args, 1 if name != 'has_ability' else 2, 2)
            self.player_arg(args[0], chunk, scope, f'{name}()')
            cat = self.cat(self.category_mask(args[1])) if len(args) == 2 else CAT_ANY
            chunk.op('HASAB' if name == 'has_ability' else 'READY', cat)
            if name == 'ready':
                chunk.op('TRUTH')
                return BOOL
            return NUM
        if name == 'in_category':
            self.nargs(node, args, 2)
            self.expect_value(args[0], self.gen(args[0], chunk, scope), 'in_category()')
            chunk.op('INCAT', self.cat(self.category_mask(args[1])))
            return BOOL
        if name in ('in_form', 'form_ability'):
            self.nargs(node, args, 1)
            self.player_arg(args[0], chunk, scope, f'{name}()')
            chunk.op('INFORM' if name == 'in_form' else 'FORMAB')
            return BOOL if name == 'in_form' else NUM
        if name == 'since_form_entry':
            self.nargs(node, args, 1)
            inner = Node('not', node.pos, Node('call', node.pos, 'in_form', [args[0]]))
            return self.gen_history(node, 'since', [inner], chunk, scope)
        if name in SLOT_KINDS:
            return self.gen_history(node, name, args, chunk, scope)
        if name in ('count', 'any', 'sources', 'min_dist', 'coverage'):
            return self.gen_aggregate(node, name, args, chunk, scope)
        if name in LOOP_MODES:
            self.fail(node, f'{name}() takes `for o in opponents: <expr>` or a collection')
        self.fail(node, f'unknown function {name!r}')

    def sub_chunk(self, label):
        chunk = Chunk(label)
        self.chunks.append(chunk)
        return chunk

    def gen_sub(self, node, label, scope, want_value=True):
        chunk = self.sub_chunk(label)
        t = self.gen(node, chunk, scope)
        if want_value:
            self.expect_value(node, t, label)
        chunk.op('END')
        return chunk

    def gen_loop(self, node, mode, args, chunk, scope):
        self.nargs(node, args, 1)
        loop = args[0]
        var, coll, body = loop.args
        if coll != 'opponents':
            self.fail(loop, f'`for {var} in {coll}`: only `opponents` can be looped over')
        if scope.domain != 'dec':
            self.fail(loop, 'a `for` loop cannot appear inside an entities/slimes/events filter')
        if scope.depth >= MAX_DEPTH:
            self.fail(loop, f'aggregates nest at most {MAX_DEPTH} deep')
        if len(scope.vars) >= MAX_VARS:
            self.fail(loop, f'at most {MAX_VARS} nested loops')
        if var in ('self', 'opp', 'ev') or var in scope.vars:
            self.fail(loop, f'loop variable {var!r} shadows a name')
        index = len(scope.vars)
        sub = self.gen_sub(body, f'{mode}(for {var} ...)', scope.child(vars={**scope.vars, var: index}, depth=scope.depth + 1))
        chunk.op('LOOP', LOOP_MODES.index(mode))
        chunk.ref(sub)
        chunk.word(index)
        return BOOL if mode in ('any', 'all') else NUM

    def gen_aggregate(self, node, name, args, chunk, scope):
        colls = [a for a in args if a.kind == 'coll']
        if len(colls) != 1:
            self.fail(node, f'{name}() needs one collection: {name}(entities where ...)')
        if scope.domain in ('ent', 'slime'):
            self.fail(node, f'{name}() cannot appear inside another entities/slimes filter')
        if scope.depth >= MAX_DEPTH:
            self.fail(node, f'aggregates nest at most {MAX_DEPTH} deep')
        coll = colls[0]
        cname, flt = coll.args
        domain = COLLECTIONS[cname]
        if domain == 'ev' and scope.event:
            self.fail(coll, f'{cname} cannot be aggregated inside an event term or events filter')
        layout = dict(count=['coll'], any=['coll'], sources=['coll'], min_dist=['coll', 'player'],
                      coverage=['player', 'num', 'sectors', 'coll'])[name]
        if len(args) != len(layout):
            self.fail(node, f'{name}() takes ({", ".join(layout)}), got {len(args)} argument(s)')
        for want, arg in zip(layout, args):
            if (want == 'coll') != (arg.kind == 'coll'):
                self.fail(arg, f'{name}(): expected {"a collection" if want == "coll" else "a " + want} here')
        allowed = dict(count=('ent', 'slime', 'ev'), any=('ent', 'slime', 'ev'), sources=('ent',),
                       min_dist=('ent', 'slime'), coverage=('ent',))[name]
        if domain not in allowed:
            self.fail(coll, f'{name}() works over {", ".join(c for c, d in COLLECTIONS.items() if d in allowed)}, not {cname}')
        sub_scope = scope.child(domain=domain, event=(domain == 'ev') or scope.event, depth=scope.depth + 1, history=False)
        if flt is None:
            flt = Node('num', coll.pos, 1.0)
        sub = self.gen_sub(flt, f'{cname} filter', sub_scope)
        extra = 0
        if name in ('count', 'any'):
            kind = {'ent': 'COUNT_ENT', 'slime': 'COUNT_SLIME', 'ev': 'COUNT_EV'}[domain]
            if domain == 'ev':
                extra = COLLECTION_KIND[cname]
        elif name == 'sources':
            kind = 'SOURCES_ENT'
        elif name == 'min_dist':
            self.player_arg(args[1], chunk, scope, 'min_dist()')
            kind = 'MINDIST_ENT' if domain == 'ent' else 'MINDIST_SLIME'
        else:
            self.player_arg(args[0], chunk, scope, 'coverage()')
            self.expect_value(args[1], self.gen(args[1], chunk, scope), 'coverage() radius')
            sectors = self.fold(args[2])
            if sectors is None or sectors != int(sectors) or not 1 <= sectors <= 32:
                self.fail(args[2], 'coverage() sectors must be a constant from 1 to 32')
            sectors = int(sectors)
            # Sector k covers angles [2*pi*k/S, 2*pi*(k+1)/S). The VM never calls a
            # transcendental: it compares a rational pseudo-angle against these
            # boundaries, the pseudo-angles of the boundary directions.
            first = len(self.consts)
            for k in range(1, sectors):
                theta = 2 * math.pi * k / sectors
                index = self.const_fresh(pseudo_angle(np.float32(math.cos(theta)), np.float32(math.sin(theta))))
                assert index == first + k - 1
            if sectors == 1:
                first = 0
            extra = sectors | (first << 8)
            kind = 'COVER_ENT'
        chunk.op('AGG', AGG[kind])
        chunk.ref(sub)
        chunk.word(extra)
        if name == 'any':
            chunk.op('CONST', self.const(0))
            chunk.op('GT')
            return BOOL
        return NUM

    def const_fresh(self, value):
        """A constant appended without deduplication (coverage boundaries are a contiguous run)."""
        self.consts.append(float(np.float32(value)))
        return len(self.consts) - 1

    def gen_history(self, node, kind, args, chunk, scope):
        if not scope.history:
            self.fail(node, f'{kind}() cannot appear inside an entities/slimes/events filter')
        windowed = kind in ('count_over', 'max_over', 'ema')
        self.nargs(node, args, 2 if windowed else 1)
        param = 0.0
        if windowed:
            param = self.fold(args[0])
            if param is None or not param > 0:
                self.fail(args[0], f'{kind}() window must be a positive constant (ticks)')
        inner = args[-1]
        key = (kind, param, repr(inner))
        if key in self.history_cache:   # the same window over the same expression shares a slot
            chunk.op('HIST', self.history_cache[key])
            return NUM
        # History is sampled once per decision for this player: the argument is
        # compiled on its own, with no event or loop variable in scope.
        saved = self.where
        sub = self.gen_sub(inner, f'{kind}() argument', Scope('dec', event=False, vars={}, depth=0))
        self.where = saved
        offset = self.hist_floats
        cap = 0
        if kind == 'prev':
            size = 3
        elif kind == 'since':
            size = 2
        elif kind == 'ema':
            size = 3
        else:
            cap = int(math.ceil(param / self.decision_ticks)) + 1
            size = 2 + 2 * cap
        self.hist_floats += size
        self.slots.append(dict(kind=SLOT_KINDS[kind], offset=offset, chunk=sub, cap=cap, param=float(np.float32(param)),
                               text=f'{kind}({", ".join([str(param)] if windowed else [])}{", " if windowed else ""}...)'))
        self.history_cache[key] = len(self.slots) - 1
        chunk.op('HIST', len(self.slots) - 1)
        return NUM

    # -- style files
    def add_style(self, path, prefix=None):
        path = Path(path)
        try:
            doc = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise CompileError(f'{path}: {error}') from None
        known = {'name', 'kits', 'fade', 'term', 'let', 'decision_ticks', 'description', 'categories', 'notes'}
        for key in doc:
            if key not in known:
                raise CompileError(f'{path}: unknown key {key!r} (known: {", ".join(sorted(known))})')
        name = doc.get('name', path.stem)
        self.decision_ticks = int(doc.get('decision_ticks', self.decision_ticks))
        self.lets, self.let_text, self.history_cache = {}, {}, {}
        for key, text in dict(doc.get('let', {})).items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key) or key in KEYWORDS:
                raise CompileError(f'{path}: let name {key!r} is not an identifier')
            self.let_text[key] = str(text)
            self.lets[key] = Parser(str(text), f'{path}: let {key}').parse()
        terms = doc.get('term', [])
        if not terms:
            raise CompileError(f'{path}: no [[term]]')
        for i, term in enumerate(terms):
            for key in term:
                if key not in ('on', 'when', 'value', 'name', 'note'):
                    raise CompileError(f'{path}: term {i}: unknown key {key!r} (known: on, when, value, name, note)')
            if 'value' not in term:
                raise CompileError(f'{path}: term {i}: `value` is required')
            on = term.get('on')
            if on is None:
                kind = 0
            elif on in EVENT_KINDS:
                kind = EVENT_KINDS[on]
            else:
                raise CompileError(f'{path}: term {i}: on = {on!r}; known events: {", ".join(EVENT_KINDS)} (omit `on` for a per-decision term)')
            tname = str(term.get('name', f'term{i}'))
            if prefix:
                tname = f'{prefix}.{tname}'
            scope = Scope('dec', event=bool(kind))
            chunks = []
            for field, default in (('when', 'true'), ('value', None)):
                text = term.get(field, default)
                if isinstance(text, bool):
                    text = 'true' if text else 'false'
                text = str(text)
                self.where, self.text = f'{path}: term {i} ({tname}) `{field}`', text
                chunks.append(self.gen_sub(Parser(text, self.where).parse(), f'{tname}.{field}', scope))
            self.terms.append(dict(kind=kind, when=chunks[0], value=chunks[1], name=tname,
                                   when_text=str(term.get('when', 'true')), value_text=str(term['value'])))
        return dict(name=name, kits=doc.get('kits'), fade=doc.get('fade'), path=str(path))

    # -- output
    def link(self):
        code, base = [], {}
        for chunk in self.chunks:
            base[id(chunk)] = len(code)
            code.extend(chunk.words)
        for chunk in self.chunks:
            b = base[id(chunk)]
            for at, target in chunk.refs:
                code[b + at] = base[id(target)]
            for at, target in chunk.jumps:
                code[b + at] |= (b + target) << 8
        return code, base


def pseudo_angle(dx, dy):
    """Monotone in the angle of (dx, dy), in [0, 4); float32 like the VM."""
    f = np.float32
    dx, dy = f(dx), f(dy)
    ax, ay = abs(dx), abs(dy)
    s = f(ax + ay)
    if s == 0:
        return f(0)
    if dy >= 0:
        return f(dy / s) if dx >= 0 else f(f(1) + f(ax / s))
    return f(f(2) + f(ay / s)) if dx < 0 else f(f(3) + f(ax / s))


def load_categories(path, abilities):
    doc = tomllib.loads(Path(path).read_text())
    ids = {canon(name): i for i, name in abilities.items()}
    out = {}
    for cat, names in doc.items():
        mask = 0
        for name in names:
            if canon(name) not in ids:
                raise CompileError(f'{path}: category {cat!r}: unknown ability {name!r}')
            mask |= 1 << ids[canon(name)]
        out[cat] = mask
    return out


class Program:
    """A compiled style: bytes plus the names and texts its dump shows."""
    def __init__(self, data, terms, styles, compiler):
        self.data, self.terms, self.styles = data, terms, styles
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.compiler = compiler

    @property
    def term_names(self):
        return [t['name'] for t in self.terms]

    @property
    def fade(self):
        return decode(self.data)['fade']

    def dump(self):
        p = decode(self.data)
        lines = [f'style program v{VERSION} sha256 {self.sha256}',
                 f'styles: {", ".join(s["name"] for s in self.styles)}  fade {p["fade"]}  '
                 f'history floats/player {p["hist_floats"]}  max stack {p["max_stack"]}  code {len(p["code"])} words']
        lines.append('constants: ' + ', '.join(f'{i}={v:g}' for i, v in enumerate(p['consts'])))
        for i, mask in enumerate(p['cats']):
            lines.append(f'category {i}: ' + ', '.join(str(b) for b in range(64) if mask >> b & 1))
        for i, s in enumerate(p['slots']):
            lines.append(f'slot {i}: {SLOT_NAMES[s["kind"]]} param {s["param"]:g} cap {s["cap"]} '
                         f'offset {s["offset"]} -> pc {s["pc"]}')
        for i, (t, meta) in enumerate(zip(p['terms'], self.terms)):
            on = {0: 'decision'} | {v: k for k, v in EVENT_KINDS.items()}
            lines.append(f'term {i} {meta["name"]} [{on[t["kind"]]}]: when pc {t["when"]} ({meta["when_text"]}); '
                         f'value pc {t["value"]} ({meta["value_text"]})')
        lines.append('code:')
        lines += disassemble(p)
        return '\n'.join(lines) + '\n'


def compile_files(paths, categories=None):
    abilities = ability_table()
    categories = load_categories(categories or ROOT / 'styles/categories.toml', abilities)
    compiler = Compiler(categories, abilities)
    styles = []
    for path in paths:
        styles.append(compiler.add_style(path, prefix=Path(path).stem if len(paths) > 1 else None))
    fade = styles[0]['fade'] or {}
    start, end = float(fade.get('start', 0.0)), float(fade.get('end', 1.0))
    if not 0 <= start < end:
        raise CompileError(f'{paths[0]}: fade needs 0 <= start < end, got {start}, {end}')
    code, base = compiler.link()
    for kind, n in (('terms', len(compiler.terms)), ('slots', len(compiler.slots)), ('consts', len(compiler.consts)),
                    ('cats', len(compiler.cats)), ('code', len(code)), ('hist_floats', compiler.hist_floats)):
        if n > LIMITS[kind]:
            raise CompileError(f'program too large: {n} {kind} (limit {LIMITS[kind]})')
    out = bytearray()
    max_stack = verify_stack(code, [base[id(c)] for c in compiler.chunks])
    out += MAGIC + struct.pack('<9I2f', VERSION, len(compiler.consts), len(code), len(compiler.slots), len(compiler.terms),
                               len(compiler.cats), compiler.hist_floats, max_stack, 0, start, end)
    out += struct.pack(f'<{len(compiler.consts)}f', *compiler.consts)
    out += struct.pack(f'<{len(compiler.cats)}Q', *compiler.cats)
    for s in compiler.slots:
        out += struct.pack('<4If', s['kind'], s['offset'], base[id(s['chunk'])], s['cap'], s['param'])
    for t in compiler.terms:
        out += struct.pack('<4I', t['kind'], base[id(t['when'])], base[id(t['value'])], 0)
    out += struct.pack(f'<{len(code)}I', *code)
    return Program(bytes(out), compiler.terms, styles, compiler)


def verify_stack(code, entries):
    """Maximum stack depth over every chunk (the C loader checks the same)."""
    deepest = 0
    for pc in entries:
        depth, at = 0, {}
        while True:
            if pc in at:
                depth = at.pop(pc)
            word = code[pc]
            name = OPS[word & 0xff]
            if name == 'END':
                if depth != 1:
                    raise CompileError(f'internal: chunk ends with stack depth {depth}')
                break
            if name in ('JZ', 'JNZ'):
                at[word >> 8] = depth
                depth -= 1
            elif name in ('AGG', 'LOOP'):
                agg = AGG_KINDS[word >> 8 & 0xffff] if name == 'AGG' else None
                pops = {'MINDIST_ENT': 1, 'MINDIST_SLIME': 1, 'COVER_ENT': 2}.get(agg, 0)
                depth += 1 - pops
            else:
                pops, pushes = EFFECT[name]
                depth += pushes - pops
            deepest = max(deepest, depth)
            pc += 1 + EXTRA.get(name, 0)
    if deepest > MAX_STACK:
        raise CompileError(f'expression too deep: stack {deepest} > {MAX_STACK}')
    return deepest


def decode(data):
    """Parse style.bin (the format bopl_style.h loads)."""
    if data[:4] != MAGIC:
        raise ValueError('not a style program')
    (version, n_consts, n_code, n_slots, n_terms, n_cats, hist_floats, max_stack, flags, start, end) = \
        struct.unpack_from('<9I2f', data, 4)
    if version != VERSION:
        raise ValueError(f'style program version {version}, expected {VERSION}')
    at = 4 + 44
    consts = list(struct.unpack_from(f'<{n_consts}f', data, at)); at += 4 * n_consts
    cats = list(struct.unpack_from(f'<{n_cats}Q', data, at)); at += 8 * n_cats
    slots = []
    for _ in range(n_slots):
        kind, offset, pc, cap, param = struct.unpack_from('<4If', data, at); at += 20
        slots.append(dict(kind=kind, offset=offset, pc=pc, cap=cap, param=param))
    terms = []
    for _ in range(n_terms):
        kind, when, value, _r = struct.unpack_from('<4I', data, at); at += 16
        terms.append(dict(kind=kind, when=when, value=value))
    code = list(struct.unpack_from(f'<{n_code}I', data, at)); at += 4 * n_code
    if at != len(data):
        raise ValueError('trailing bytes in style program')
    return dict(consts=consts, cats=cats, slots=slots, terms=terms, code=code, hist_floats=hist_floats,
                max_stack=max_stack, fade=(start, end))


def disassemble(p):
    code, lines, pc = p['code'], [], 0
    fields = dict(HEADER=HEADER_FIELDS, PFIELD=PLAYER_FIELDS, BFIELD=SLIME_FIELDS, BFIELDI=SLIME_ARRAYS,
                  EVFIELD=EVENT_FIELDS, EFIELD=ENTITY_FIELDS, SFIELD=SLIME_FIELDS, SFIELDI=SLIME_ARRAYS,
                  TELEM=TELEMETRY_FIELDS)
    while pc < len(code):
        word = code[pc]
        name, arg = OPS[word & 0xff], word >> 8
        text = name
        if name == 'CONST':
            text += f' {p["consts"][arg]:g}'
        elif name in fields:
            text += f' {fields[name][arg]}'
        elif name in ('JZ', 'JNZ'):
            text += f' -> {arg}'
        elif name in ('READY', 'HASAB', 'INCAT'):
            text += ' any' if arg == CAT_ANY else f' cat{arg}'
        elif name in ('VAR', 'HIST'):
            text += f' {arg}'
        elif name == 'AGG':
            text += f' {AGG_KINDS[arg]} filter@{code[pc + 1]} extra {code[pc + 2]}'
        elif name == 'LOOP':
            text += f' {LOOP_MODES[arg]} body@{code[pc + 1]} var {code[pc + 2]}'
        lines.append(f'  {pc:5d}  {text}')
        pc += 1 + EXTRA.get(name, 0)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('styles', nargs='+', type=Path)
    parser.add_argument('-o', '--output', type=Path, help='write the program here (and <output>.txt, the dump)')
    parser.add_argument('--categories', type=Path)
    parser.add_argument('--dump', action='store_true', help='print the readable dump')
    args = parser.parse_args()
    try:
        program = compile_files(args.styles, args.categories)
    except CompileError as error:
        print(f'style error: {error}', file=sys.stderr)
        sys.exit(1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(program.data)
        args.output.with_suffix(args.output.suffix + '.txt').write_text(program.dump())
    if args.dump or not args.output:
        print(program.dump(), end='')
    print(f'{len(program.data)} bytes, sha256 {program.sha256}', file=sys.stderr)


if __name__ == '__main__':
    main()
