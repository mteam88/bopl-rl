"""Kit pinning for frozen opponents (train.py --pinned-opponents; docs/LEAGUE.md "Kit pinning").

A frozen bank's opponent normally plays whatever kit the run deals its seat, so
an exploiter or specialist with kits of its own is handed the learner's draw and
plays off-kit. With pinning, a bank environment deals its frozen seat one of the
rows installed for the member currently loaded in its bank: [member kit |
learner kit], where a learner half of 0,0,0 keeps the learner's own draw. A
scripted bot on such a bank plays its own kit and never a space map (that match
goes to the bank's frozen weights). The C side is bopl_puffer.h
bopl_bank_kit_rows; this module installs rows and audits what was dealt.
"""
import ctypes
from collections import Counter

import numpy as np

ROWS_MAX = 64          # BOPL_BANK_KITS_MAX
LAYOUT_STRIDE = 22     # bopl_vec_layout: 8 + BOPL_MAX_SLOTS


def library(module):
    lib = ctypes.CDLL(module.__file__)
    lib.bopl_bank_kits_install.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.bopl_bank_kits_install.restype = ctypes.c_int
    lib.bopl_bot_kit_install.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.bopl_bot_kit_install.restype = ctypes.c_int
    lib.bopl_bank_kit_redraw_count.restype = ctypes.c_ulong
    for name in ('bopl_vec_layout', 'bopl_vec_kits', 'bopl_vec_pins'):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    return lib


def member_rows(kits):
    """A member's declared kits (registry.kits) as rows: its kit, the learner's own draw."""
    rows = [[int(x) for x in kit] + [0, 0, 0] for kit in kits]
    if not 1 <= len(rows) <= ROWS_MAX or any(len(r) != 6 for r in rows):
        raise ValueError(f'a pinned member needs 1..{ROWS_MAX} three-ability kits, got {kits}')
    return rows


def install(lib, bank, rows):
    """Install bank `bank`'s rows ([] = the run's own draw); returns the table generation."""
    flat = (ctypes.c_int * max(1, 6 * len(rows)))(*[int(x) for row in rows for x in row])
    gen = lib.bopl_bank_kits_install(bank, flat, len(rows))
    if gen < 0:
        raise ValueError(f'bopl_bank_kits_install refused bank {bank} rows {rows}')
    return gen


def install_bot_kit(lib, kit):
    if lib.bopl_bot_kit_install((ctypes.c_int * 3)(*[int(x) for x in kit])):
        raise ValueError(f'bopl_bot_kit_install refused {kit}')


class PinnedBackend:
    """The trainer backend native_selfplay swaps banks through, with kit pinning.

    load_frozen_bank installs the bank's rows for the member it is about to load
    (rows_by_path; any other checkpoint gets none) and then loads the weights,
    both between rollouts. Everything else is the module's.
    """
    def __init__(self, module, lib, rows_by_path):
        self.module, self.lib, self.rows_by_path = module, lib, dict(rows_by_path)
        self.loaded, self.gen = {}, {}

    def load_frozen_bank(self, trainer, bank, path):
        self.gen[bank] = install(self.lib, bank, self.rows_by_path.get(str(path), []))
        self.loaded[bank] = str(path)
        return self.module.load_frozen_bank(trainer, bank, path)

    def __getattr__(self, name):
        return getattr(self.module, name)


def read(lib, obs_ptr, envs):
    """(layout, kits, pins) of every environment, as bopl_vec_layout / _kits / _pins arrays."""
    lay = np.empty((envs, LAYOUT_STRIDE), dtype=np.int32)
    kits = np.empty((envs, 6), dtype=np.int32)
    pins = np.empty((envs, 4), dtype=np.int32)
    p = ctypes.c_void_p(obs_ptr)
    for fn, arr in ((lib.bopl_vec_layout, lay), (lib.bopl_vec_kits, kits), (lib.bopl_vec_pins, pins)):
        if fn(p, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), arr.size) != envs:
            raise RuntimeError(f'{fn.__name__}: environment count changed')
    return lay, kits, pins


def frozen_matches(lay, kits, pins):
    """Per bank environment: (env, bank, match, side, learner kit, frozen kit, map, bot kind, pin src, pin gen)."""
    out = []
    for i in np.nonzero(lay[:, 2])[0]:
        side = int(lay[i, 4])
        out.append((int(i), int(lay[i, 5]), int(lay[i, 6]), side,
                    tuple(int(x) for x in kits[i, side * 3:side * 3 + 3]),
                    tuple(int(x) for x in kits[i, (1 - side) * 3:(1 - side) * 3 + 3]),
                    int(pins[i, 0]), int(pins[i, 1]), int(pins[i, 2]), int(pins[i, 3])))
    return out


class Audit:
    """The kits each bank's frozen seat holds, read between rollouts (no env thread runs).

    Per bank: the member loaded, how many of its environments' current matches
    were pinned under the current table (member), dealt a bot kit (bot), started
    before the last swap (earlier), and how many broke the rule (off_kit: a
    current-table match whose frozen kit is not the member's, or a bot match off
    the bot kit or on a space map). off_kit must stay 0.
    """
    def __init__(self, lib, obs_ptr, envs, backend, kits_by_path, names_by_path, bot_kit, space_maps):
        self.lib, self.obs_ptr, self.envs, self.backend = lib, obs_ptr, envs, backend
        self.kits = {path: {tuple(k) for k in kits} for path, kits in kits_by_path.items()}
        self.names, self.bot_kit, self.space = names_by_path, tuple(bot_kit) if bot_kit else None, set(space_maps)

    def __call__(self):
        lay, kits, pins = read(self.lib, self.obs_ptr, self.envs)
        banks = {}
        for _, bank, _, _, _, frozen, map_id, bot, src, gen in frozen_matches(lay, kits, pins):
            path = self.backend.loaded.get(bank)
            row = banks.setdefault(bank, dict(member=self.names.get(path, path and path.rsplit('/', 1)[-1]),
                                              pinned=path in self.kits, member_matches=0, bot=0, earlier=0,
                                              unpinned=0, off_kit=0, kits=Counter()))
            if bot:
                row['bot'] += 1
                row['off_kit'] += (self.bot_kit is not None and frozen != self.bot_kit) or map_id in self.space
            elif gen != self.backend.gen.get(bank):
                row['earlier'] += 1
            elif path in self.kits:
                row['member_matches'] += 1
                row['off_kit'] += frozen not in self.kits[path] or src != 1
            else:
                row['unpinned'] += 1
                row['off_kit'] += src != 0
            row['kits']['-'.join(map(str, frozen))] += 1
        for row in banks.values():
            row['kits'] = dict(row['kits'].most_common(6))
        return dict(banks={str(b): banks[b] for b in sorted(banks)},
                    redraws=int(self.lib.bopl_bank_kit_redraw_count()))
