"""C/Python agreement for style programs on the replay dumps.

  python eval/style_check.py [--lib native/artifacts/rl/libstyle_check.so] [--games 200] styles/*.toml eval/style_check.toml

Compiles the given styles into one program, then for each dump rebuilds every
decision's BoplRlState rows (Q32.32 from the dumps' float32, exact: each float32
is the int64 rounded to 24 bits) and runs the C VM (train/bopl_style.h via
eval/style_check.c) next to eval/style_score.py's numpy interpreter. Every
decision's per-term payout and total must agree bit for bit. Each game runs
twice: as dumped (no events, event_support 0) and with synthetic events derived
from the dump's death/kill counters (event_support 1), which drives the event
terms, event aggregates and ev.* fields through both interpreters.
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'eval'))
import style_compile as sc  # noqa: E402
import style_score as ss  # noqa: E402

U4, I8 = '<u4', '<i8'
PLAYER_DT = np.dtype([(n, U4) for n in ('id', 'team', 'flags', 'alive_bodies', 'stocks', 'kills', 'deaths', 'cause_of_death')]
                     + [(n, I8) for n in ('x', 'y', 'scale', 'invisible_time', 'ability_time')])
TELEM_DT = np.dtype([(n, U4) for n in sc.TELEMETRY_FIELDS])
ENTITY_DT = np.dtype([(n, U4) for n in ('id', 'kind', 'layer', 'owner', 'flags', 'hierarchy', 'source_ability', 'timer')]
                     + [(n, I8) for n in ('x', 'y', 'vx', 'vy', 'right_x', 'right_y', 'up_x', 'up_y', 'radius', 'rotation',
                                          'angular_velocity', 'inverse_mass', 'clock_rate')])
SLIME_DT = np.dtype([(n, U4) for n in ('id', 'owner', 'flags', 'state')] + [(n, I8) for n in ('x', 'y', 'vx', 'vy', 'rotation', 'scale')]
                    + [('abilities', U4, (3,)), ('ability_flags', U4, (3,)), ('cooldowns', I8, (3,)), ('timers', I8, (3,))])
EVENT_DT = np.dtype([(n, U4) for n in ('kind', 'tick', 'subject', 'actor', 'entity', 'source_ability', 'cause', 'flags',
                                        'related', 'related_actor', 'related_tick', 'creator')]
                    + [(n, I8) for n in ('x', 'y', 'vx', 'vy', 'ax', 'ay', 'avx', 'avy')])
Q_FIELDS = {'x', 'y', 'vx', 'vy', 'ax', 'ay', 'avx', 'avy', 'scale', 'invisible_time', 'ability_time', 'right_x', 'right_y',
            'up_x', 'up_y', 'radius', 'rotation', 'angular_velocity', 'inverse_mass', 'clock_rate', 'cooldowns', 'timers'}


def q(values):
    """float32 -> the Q32.32 int64 it was rounded from (exact)."""
    out = (np.asarray(values, np.float64) * 2.0**32).astype(np.int64)
    back = (out.astype(np.float32) * np.float32(2.0**-32)).astype(np.float32)
    if not np.array_equal(back.view(np.uint32), np.asarray(values, np.float32).view(np.uint32)):
        raise ValueError('a dump value does not round-trip through Q32.32')
    return out


def pack(table, dtype):
    out = np.zeros(len(table), dtype)
    for name in dtype.names:
        if name not in table.dtype.names:
            continue
        out[name] = q(table[name]) if name in Q_FIELDS else table[name]
    return out


def synthetic_events(game):
    """DEATH events where a player's death counter rises (actor: an opponent whose kills rose), plus a BLINK every 97 decisions."""
    deaths = game.player_raw['deaths'].astype(np.int64)
    kills = game.player_raw['kills'].astype(np.int64)
    env = game.telem_raw[:, :, sc.TELEMETRY_FIELDS.index('deaths_environment')].astype(np.int64)
    rows = []

    def body_xy(d, p):
        r = game.body[d, p]
        if r < 0:
            return game.player['x'][d, p], game.player['y'][d, p], np.float32(0), np.float32(0)
        return game.sl_f['x'][r], game.sl_f['y'][r], game.sl_f['vx'][r], game.sl_f['vy'][r]

    for d in range(1, game.D):
        tick = int(game.ticks[d])
        for p in range(game.P):
            if deaths[d, p] <= deaths[d - 1, p]:
                continue
            actor = 0
            for k in range(game.P):
                if k != p and kills[d, k] > kills[d - 1, k]:
                    actor = k + 1
                    break
            x, y, vx, vy = body_xy(d, p)
            ax, ay, avx, avy = body_xy(d, actor - 1) if actor else (0, 0, 0, 0)
            source = 0
            if actor and game.body[d, actor - 1] >= 0:
                r = game.body[d, actor - 1]
                form = np.flatnonzero(game.sl_aflags[r] & 8)
                source = int(game.sl_abilities[r][form[0] if len(form) else 0])
            rows.append((tick - (d % 3), 1, p + 1, actor, 1000 + d, source, 7 if env[d, p] > env[d - 1, p] else 2,
                         (4 if d % 2 else 0) | (1 if d % 3 == 0 else 0) | 2, 500 + p, actor if d % 2 else 0, tick - 9,
                         x, y, vx, vy, ax, ay, avx, avy))
        if d % 97 == 0:
            x, y, vx, vy = body_xy(d, 0)
            rows.append((tick, 2, 2 if game.P > 1 else 1, 1, 77, 6, 0, 1, 0, 0, 0, x, y, vx, vy, 0, 0, 0, 0))
    events = np.array(rows, dtype=ss.EVENT_DTYPE) if rows else np.zeros(0, ss.EVENT_DTYPE)
    order = np.argsort(events['tick'], kind='stable')
    return events[order]


def c_events(events):
    out = np.zeros(len(events), EVENT_DT)
    for name in EVENT_DT.names:
        if name == 'creator':
            continue
        out[name] = q(events[name]) if name in Q_FIELDS else events[name]
    return out


def run_c(lib, data, game, events, support, n_terms):
    D, P = game.D, game.P
    hdr = np.zeros((D, 6), np.uint32)
    for k, name in enumerate(('tick', 'episode_ticks', 'flags', 'entity_count', 'slime_count', 'sudden_death')):
        hdr[:, k] = game.header[name]
    if not (np.array_equal(hdr[:, 3], game.ent_count) and np.array_equal(hdr[:, 4], game.sl_count)):
        raise ValueError('header counts disagree with the dumped rows')
    players = np.zeros((D, P), PLAYER_DT)
    for name in PLAYER_DT.names:
        raw = game.player_raw[name]
        players[name] = q(raw) if name in Q_FIELDS else raw
    telem = np.zeros((D, P), TELEM_DT)
    for k, name in enumerate(TELEM_DT.names):
        telem[name] = game.telem_raw[:, :, k]
    ents = pack(game.ent, ENTITY_DT)
    slimes = pack(game.sl, SLIME_DT)
    ent_off = np.concatenate([[0], np.cumsum(game.ent_count)]).astype(np.int64)
    sl_off = np.concatenate([[0], np.cumsum(game.sl_count)]).astype(np.int64)
    if events is not None and len(events):
        evc = c_events(events)
        dec = np.searchsorted(game.ticks, events['tick'].astype(np.int64), side='left')
        keep = dec < D
        evc, dec = evc[keep], dec[keep]
        ev_off = np.concatenate([[0], np.cumsum(np.bincount(dec, minlength=D))]).astype(np.int64)
        ev_ptr = evc.ctypes.data
    else:
        evc, ev_off, ev_ptr = None, np.zeros(D + 1, np.int64), None
    terms = np.zeros((P, D, n_terms), np.float32)
    totals = np.zeros((P, D), np.float32)
    error = ctypes.create_string_buffer(256)
    rc = lib.style_check_run(data, len(data), D, P, hdr.ctypes.data, players.ctypes.data, telem.ctypes.data,
                             ents.ctypes.data if len(ents) else None, ent_off.ctypes.data,
                             slimes.ctypes.data if len(slimes) else None, sl_off.ctypes.data,
                             ev_ptr, ev_off.ctypes.data, int(support), terms.ctypes.data, totals.ctypes.data, error, 256)
    if rc:
        raise RuntimeError(f'style_check_run {rc}: {error.value.decode()}')
    return terms, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('styles', nargs='+', type=Path)
    parser.add_argument('--lib', type=Path, default=ROOT / 'native/artifacts/rl/libstyle_check.so')
    parser.add_argument('--dumps', type=Path, default=ROOT / 'native/artifacts/human-replays/dump')
    parser.add_argument('--games', type=int, default=0, help='first N dumps (0: all)')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.lib))
    lib.style_check_run.restype = ctypes.c_int
    lib.style_check_run.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*9 \
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    sizes = (ctypes.c_int*5)()
    lib.style_check_sizes(sizes)
    assert list(sizes) == [d.itemsize for d in (PLAYER_DT, TELEM_DT, ENTITY_DT, SLIME_DT, EVENT_DT)], list(sizes)
    program = sc.compile_files(args.styles)
    data = program.data
    prog = ss.Decoded(data)
    n_terms = len(prog.terms)
    files = sorted(args.dumps.glob('*.npz'), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.games:
        files = files[:args.games]
    report = dict(styles=[str(s) for s in args.styles], sha256=program.sha256, terms=program.term_names, games=0,
                  decisions=0, values=0, mismatches=0, events=0, first_mismatch=None,
                  c_totals=[0.0]*n_terms, py_totals=[0.0]*n_terms, nonzero=[0]*n_terms)
    started = time.time()
    for path in files:
        game = ss.Game(path)
        if game.D == 0:
            continue
        events = synthetic_events(game)
        for support, evs in ((0, None), (1, events)):
            game.set_events(evs)
            c_terms, c_totals = run_c(lib, data, game, evs, support, n_terms)
            for p in range(game.P):
                pays, total = ss.Player(prog, game, p, support).run()
                a = np.concatenate([c_terms[p].view(np.uint32).ravel(), c_totals[p].view(np.uint32)])
                b = np.concatenate([pays.view(np.uint32).ravel(), total.view(np.uint32)])
                bad = np.flatnonzero(a != b)
                report['values'] += len(a)
                report['mismatches'] += len(bad)
                if len(bad) and report['first_mismatch'] is None:
                    i = int(bad[0])
                    d, t = (i // n_terms, i % n_terms) if i < game.D * n_terms else (i - game.D * n_terms, 'total')
                    report['first_mismatch'] = dict(file=path.name, support=support, player=p + 1, decision=int(d),
                                                    tick=int(game.ticks[d]), term=t if t == 'total' else program.term_names[t],
                                                    c=float(a[i:i+1].view(np.float32)[0]), py=float(b[i:i+1].view(np.float32)[0]))
                ct = c_terms[p].astype(np.float64).sum(0); pt = pays.astype(np.float64).sum(0)
                report['c_totals'] = [x + y for x, y in zip(report['c_totals'], ct.tolist())]
                report['py_totals'] = [x + y for x, y in zip(report['py_totals'], pt.tolist())]
                report['nonzero'] = [x + int(y) for x, y in zip(report['nonzero'], (pays != 0).sum(0))]
            report['decisions'] += game.D * game.P
            report['events'] += 0 if evs is None else len(evs)
        report['games'] += 1
    report['seconds'] = round(time.time() - started, 1)
    report['agree'] = report['mismatches'] == 0
    text = json.dumps(report, indent=1)
    if args.json:
        args.json.write_text(text)
    print(text)
    print('STYLE AGREEMENT', 'PASS' if report['agree'] else 'FAIL')
    sys.exit(0 if report['agree'] else 1)


if __name__ == '__main__':
    main()
