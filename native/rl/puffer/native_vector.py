"""Validated learner/frozen routing for one native PufferLib trainer."""
import ctypes as C

import numpy as np

import counters


class NativeVector:
    def __init__(self, backend, workers, buffers, banks):
        if workers < 2 or workers % (2 * buffers) or not 2 <= banks <= 8:
            raise ValueError('Each buffer requires complete self-play/historical pairs and 2..8 banks')
        self.backend, self.workers, self.buffers, self.banks = backend, workers, buffers, banks
        self.pairs = workers // (2 * buffers)
        self.agents_per_buffer = self.pairs * (3 + banks)
        self.total_agents = self.agents_per_buffer * buffers
        self.learner_rows = workers * 3 // 2
        self.lib = counters.bind(C.CDLL(backend.__file__))
        for name, types in {
            'bopl_vec_handles': [C.POINTER(C.c_void_p), C.c_int],
            'bopl_vec_reset': [C.c_void_p],
            'bopl_vec_layout': [C.c_void_p, C.POINTER(C.c_int), C.c_int],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types, C.c_int
        self.lib.bopl_set_valid_banks.argtypes = [C.c_uint]
        self.set_valid_banks(1)

    def set_valid_banks(self, mask):
        if not 0 < mask < (1 << self.banks):
            raise ValueError('Frozen draw mask must select at least one configured bank')
        self.lib.bopl_set_valid_banks(mask)

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
        expected = [[2, 2, 0], [1 + self.banks, 2, self.banks]] * (self.workers // 2)
        if before[:, :3].tolist() != expected:
            raise RuntimeError('Expected alternating self-play and historical games')
        perm = np.full(self.total_agents, -1, dtype=np.int32)
        slot = 0
        for buffer in range(self.buffers):
            primary, frozen = [], [[] for _ in range(self.banks)]
            for _ in range(self.pairs):
                primary.extend((slot, slot + 1, slot + 2))
                for bank in range(self.banks):
                    frozen[bank].append(slot + 3 + bank)
                slot += 3 + self.banks
            order = primary + [row for group in frozen for row in group]
            perm[order] = np.arange(self.agents_per_buffer) + buffer * self.agents_per_buffer
        if sorted(perm.tolist()) != list(range(self.total_agents)):
            raise RuntimeError('Invalid native agent permutation')
        self.backend.set_agent_perm(trainer, perm)
        after, slot = self.layout(), 0
        for row in after:
            size = int(row[0])
            if row[8:8+size].tolist() != perm[slot:slot+size].tolist():
                raise RuntimeError('Native environment did not install the trainer permutation')
            slot += size

    def reset(self):
        if self.lib.bopl_vec_reset(C.c_void_p(self.gpu_obs_ptr)):
            raise RuntimeError('Native post-routing reset failed')

    def bank_references(self):
        references = [0] * self.banks
        for row in self.layout():
            if row[2]:
                bank = int(row[5])
                if not 0 <= bank < self.banks:
                    raise RuntimeError('Historical match references an invalid bank')
                references[bank] += 1
        return references

    def snapshot(self):
        return counters.snapshot(self.lib, self)
