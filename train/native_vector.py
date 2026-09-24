"""Validated learner/frozen routing for one native PufferLib trainer."""
import ctypes as C

import numpy as np

import counters


class NativeVector:
    def __init__(self, backend, workers, buffers, banks):
        if workers < 2 or workers % (2 * buffers) or not 2 <= banks <= 8:
            raise ValueError('Each buffer requires complete self-play/historical pairs and 2..8 banks')
        if (workers // 2) % banks:
            raise ValueError('Historical games must be dealt evenly to the banks')
        self.backend, self.workers, self.buffers, self.banks = backend, workers, buffers, banks
        self.pairs = workers // (2 * buffers)
        self.agents_per_buffer = self.pairs * (3 + banks)
        self.total_agents = self.agents_per_buffer * buffers
        self.learner_rows = workers * 3 // 2
        self.historical_per_bank = workers // 2 // banks
        # Compact row layout (THROUGHPUT.md 3.C): a historical environment's
        # opponent plays on one of its `banks` slots for the environment's whole
        # life, so 3 of 4 opponent rows never hold a decision. When every
        # buffer deals its historical games evenly to the banks, each buffer's
        # rows are packed [learner rows | bank 0 live rows | ... | bank n-1
        # live rows | idle rows]; the trainer's bank_layout covers the live
        # prefix and the idle tail is never uploaded, run through the policy or
        # sampled. Otherwise (pairs % banks != 0) the legacy layout puts every
        # opponent row of bank b in bank b's slice.
        self.compact = self.pairs % banks == 0
        self.primary_rows = 3 * self.pairs
        self.live_per_bank = self.pairs // banks if self.compact else self.pairs
        self.live_rows = self.primary_rows + banks * self.live_per_bank
        self.lib = counters.bind(C.CDLL(backend.__file__))
        for name, types in {
            'bopl_vec_handles': [C.POINTER(C.c_void_p), C.c_int],
            'bopl_vec_reset': [C.c_void_p],
            'bopl_vec_layout': [C.c_void_p, C.POINTER(C.c_int), C.c_int],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types, C.c_int

    def layout(self):
        out = np.empty((self.workers, 22), dtype=np.int32)
        count = self.lib.bopl_vec_layout(C.c_void_p(self.gpu_obs_ptr),
                                        out.ctypes.data_as(C.POINTER(C.c_int)), out.size)
        if count != self.workers:
            raise RuntimeError('Native environment count changed')
        return out

    def attach(self, trainer):
        handles = (C.c_void_p * 1)()
        if self.lib.bopl_vec_handles(handles, 1) != 1 or self.backend.num_envs(trainer) != self.workers:
            raise RuntimeError('Training requires exactly one vector with the requested game count')
        self.gpu_obs_ptr = handles[0]
        before = self.layout()
        # Historical game j (every second environment) is pinned to bank
        # j % banks and carries PufferLib's selfplay tag 1 + bank.
        expected = []
        for j in range(self.workers // 2):
            bank = j % self.banks
            expected += [[2, 2, 0, 0, -1], [1 + self.banks, 2, self.banks, 1 + bank, bank]]
        if before[:, [0, 1, 2, 3, 5]].tolist() != expected:
            raise RuntimeError('Expected alternating self-play and bank-pinned historical games')
        perm = np.full(self.total_agents, -1, dtype=np.int32)
        slot = 0
        for buffer in range(self.buffers):
            primary, frozen, idle = [], [[] for _ in range(self.banks)], []
            for pair in range(self.pairs):
                primary.extend((slot, slot + 1, slot + 2))
                active = int(before[2 * (buffer * self.pairs + pair) + 1, 5])
                for bank in range(self.banks):
                    (frozen[bank] if not self.compact or bank == active else idle).append(slot + 3 + bank)
                slot += 3 + self.banks
            if self.compact and any(len(rows) != self.live_per_bank for rows in frozen):
                raise RuntimeError('Historical games are not dealt evenly to the banks within a buffer')
            order = primary + [row for group in frozen for row in group] + idle
            perm[order] = np.arange(self.agents_per_buffer) + buffer * self.agents_per_buffer
        if sorted(perm.tolist()) != list(range(self.total_agents)):
            raise RuntimeError('Invalid native agent permutation')
        self.backend.set_agent_perm(trainer, perm)
        self.backend.set_env_tags(trainer, np.ascontiguousarray(before[:, 3], dtype=np.int32))
        after, slot = self.layout(), 0
        for row in after:
            size = int(row[0])
            if row[8:8+size].tolist() != perm[slot:slot+size].tolist():
                raise RuntimeError('Native environment did not install the trainer permutation')
            slot += size

    def reset(self):
        if self.lib.bopl_vec_reset(C.c_void_p(self.gpu_obs_ptr)):
            raise RuntimeError('Native post-routing reset failed')

    def snapshot(self):
        return counters.snapshot(self.lib, self)
